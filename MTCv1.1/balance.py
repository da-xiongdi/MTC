
from CoolProp.CoolProp import PropsSI
import kinetics as kn

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
