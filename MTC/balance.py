import json

import matplotlib.pyplot as plt
import pandas as pd
from CoolProp.CoolProp import PropsSI
import kinetics as kn
import numpy as np
import scipy.integrate

R = 8.314


def mass_balance(T, P, F_dict, feed_data, chem_data):
    dF_react = kn.rate(T, P, F_dict, feed_data, chem_data).loc['global']
    return dF_react


def energy_balance(T, P, F_dict, feed_data, chem_data):
    """
    calculate the energy balance in the reactor
    :param T: operating temperature, K
    :param P: operating pressure, bar
    :param F_dict: molar flow rate of each component, mol/s; dict
    :param feed_data: initial condition including space velocity, temperature, pressure; list
    :param chem_data: prescribed parameter of components and reactions; dict
    :return: temperature variation of gas
    """
    Ft = 0
    for comp in F_dict:
        Ft += comp

    # calculate the correction to volumetric flow rate
    v0 = feed_data["Sv"]  # initial volumetric flow rate (m3/s)
    T0 = feed_data["T"]  # initial temperature (K)
    P0 = feed_data["P"]  # initial pressure (bar)
    Ft0 = feed_data["Ft0"]  # initial total molar flow rate (mol/s)
    v = v0 * (P0 / P) * (T / T0) * (Ft / Ft0)  # corrected volumetric flow rate (m3/s)

    # calculate the change of the molar flow rate due to reactions
    dF_react = kn.rate(T, P, F_dict, feed_data, chem_data)
    # print(dF_react)

    # calculate the change of enthalpy due to reaction, kJ/(kg_cat s)
    dH = 0
    for react in chem_data["reaction_list"].keys():
        dH += dF_react.loc[react, 'CO2'] * float(chem_data["heat_reaction"][react])

    # calculate the heat capacity of each component, cp*n, J/(s K)
    heat_capacity = 0
    for comp in chem_data["comp_list"]:
        pi = F_dict[comp] * R * T / v
        cp = PropsSI('CPMOLAR', 'T', T, 'P', pi, comp) if pi > 0 else 0  # J/(mol K)
        heat_capacity += cp * F_dict[comp]

    dT = dH * 1e3 / heat_capacity  # K/kg_cat

    return dT


def balance(T, P, F_dict, feed_data, chem_data):
    """
    calculate the energy balance in the reactor
    :param T: operating temperature, K
    :param P: operating pressure, bar
    :param F_dict: molar flow rate of each component, mol/s; dict
    :param feed_data: initial condition including space velocity, temperature, pressure; list
    :param chem_data: prescribed parameter of components and reactions; dict
    :return: temperature and molar flux variation of gas
    """
    Ft = 0
    for comp in F_dict:
        Ft += comp

    # calculate the correction to volumetric flow rate
    v0 = feed_data["Sv"]  # initial volumetric flow rate (m3/s)
    T0 = feed_data["T"]  # initial temperature (K)
    P0 = feed_data["P"]  # initial pressure (bar)
    Ft0 = feed_data["Ft0"]  # initial total molar flow rate (mol/s)
    v = v0 * (P0 / P) * (T / T0) * (Ft / Ft0)  # corrected volumetric flow rate (m3/s)

    # calculate the change of the molar flow rate due to reactions
    dF_react = kn.rate(T, P, F_dict, feed_data, chem_data)
    # print(dF_react)

    # calculate the change of enthalpy due to reaction, kJ/(kg_cat s)
    dH = 0
    for react in chem_data["reaction_list"].keys():
        dH += dF_react.loc[react, 'CO2'] * float(chem_data["heat_reaction"][react])

    # calculate the heat capacity of each component, cp*n, J/(s K)
    heat_capacity = 0
    for comp in chem_data["comp_list"]:
        pi = F_dict[comp] * R * T / v
        cp = PropsSI('CPMOLAR', 'T', T, 'P', pi, comp) if pi > 0 else 0  # J/(mol K)
        heat_capacity += cp * F_dict[comp]

    dT = dH * 1e3 / heat_capacity  # K/kg_cat

    return dT, dF_react.loc['global']


def ode_ivp(temprature, press, feed, feed_cond, chem):
    """
    ode for the concentration distribution along the channel
    radial distribution
    the origin is located at the center of the circle
    :param inner_cond: temperature, molar fraction, and radius at inside;list
    :param outer_cond: temperature, molar fraction, and radius at outside; list
    :param P: pressure of mixture
    :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
    :return: concentration and its slop
    """

    # function

    def model(z, y):
        # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, T]
        F_in = pd.Series(np.array(y[:-1]), index=chem["comp_list"])
        temp = balance(temprature, press, F_in, feed_cond, chem)
        dF_dz = temp[1] * np.pi * 1 ** 2 / 4
        dT_dz = temp[0]
        dF_dz_res = dF_dz.values

        return np.append(dF_dz_res, [dT_dz])

    z_span = [0, 1]
    feed = feed.values
    ic = np.append(feed, [temprature])
    print(ic)
    res = scipy.integrate.solve_ivp(model, z_span, ic, method='RK45', t_eval=np.arange(0, 1, 0.01))
    t = res.t
    print(t)
    data = res.y
    print(data.shape)
    plt.plot(t, data[-1, :])
    plt.show()
    r = (data[0,-1]-data[0,0])/data[0,0]
    print(r)

    # return ysol


F = {"CO2": 1954.833424, "H2": 5864.500273, "Methanol": 0, "H2O": 0, "CO": 0}
feed_info = {'T': 483, 'P': 30, 'Sv': 10.9, 'Ft0': 7819.333697119664}
F = pd.Series(F, index=F.keys())
f1 = open('in_chem.json')
chem_dict = json.load(f1)
ode_ivp(483, 30, F, feed_info, chem_dict)
