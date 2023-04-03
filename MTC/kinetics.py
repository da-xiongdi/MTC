import numpy as np
import json

import pandas as pd

R = 8.314


def kad(T, in_dict):
    """
    calculate the equilibrium constant of adsorption
    :param T: operating temperature
    :param in_dict: prescribed chemical parameter
    :return: equilibrium constant of adsorption, 1/bar
    """
    adsorption_eq_constant = dict()
    for key, value in in_dict["kad"].items():
        adsorption_eq_constant[key] = value[0] * np.exp(value[1] / T / R)
    return adsorption_eq_constant


def keq(T, in_dict):
    """
    calculate the equilibrium constant
    :param T: operating temperature
    :param in_dict: prescribed chemical parameter
    :return: equilibrium constant
    """
    react_eq_constant = dict()
    for key, value in in_dict["keq"].items():
        # react_eq_constant[key] = 10 ** (value[0] / T - value[1])
        react_eq_constant[key] = np.exp(value[0]/T + value[1])
    # print(react_eq_constant)
    return react_eq_constant


def kr(T, in_dict):
    """
    calculate the reaction rate constant
    :param T: operating temperature, K
    :param in_dict: prescribed chemical parameter
    :return: the reaction rate constant, mol kg−1 s−1 bar-1/2
    """
    react_rate_constant = dict()
    for key, value in in_dict["kr"].items():
        react_rate_constant[key] = value[0] * np.exp(value[1] / T / R)
    return react_rate_constant


def rate(T, P, F_dict, feed_data, chem_dict):
    """
    calculate the reaction rate, mol/(kg s), per kg catalyst
    :param T: operating temperature
    :param P: operating pressure
    :param F_dict: molar flow rate of each component, mol/s; Dataframe
    :param feed_data: initial condition including space velocity, temperature, pressure; dict
    :param chem_dict: prescribed parameter of components and reactions; dict
    :return: reaction rate of each component for each and all reaction; pd.Dataframe[react*comp]
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

    # calculate the partial pressure
    Pi = dict()
    for comp in F_dict.index:
        Pi[comp] = F_dict[comp] * R * T / v
        Pi[comp] /= 1e5  # convert pa to bar
    # print(Pi)
    rate_const = kr(T, chem_dict)
    ad_const = kad(T, chem_dict)
    eq_const = keq(T, chem_dict)
    # eq_const['1'] = eq_const['2'] * eq_const['3']

    # calculate the rate of each reaction
    react_rate = dict()
    driving = rate_const['1'] * (Pi['CO2'] * Pi['H2'] - Pi['H2O'] * Pi["Methanol"] / Pi["H2"] ** 2 / eq_const['1'])
    inhibiting = (1 + ad_const["H2O/H2"] * Pi['H2O'] / Pi['H2'] +
                  ad_const["H2"] * Pi["H2"] ** 0.5 + ad_const["H2O"] * Pi["H2O"])
    react_rate['1'] = driving / inhibiting

    driving = rate_const['2'] * (Pi['CO2'] - Pi['H2O'] * Pi["CO"] / Pi["H2"] * eq_const['2'])
    react_rate['2'] = driving / inhibiting

    react_rate['3'] = 0

    # print(react_rate)
    temp = np.zeros((3, 5))
    for i in range(3):
        key = str(i + 1)
        temp[i] = np.array(react_rate[key]) * np.array(chem_dict["stoichiometry"][key])
    react_comp_rate = pd.DataFrame(temp, index=chem_dict["stoichiometry"].keys(), columns=chem_dict["comp_list"])
    react_comp_rate.loc['global'] = react_comp_rate.sum(axis=0)
    return react_comp_rate

# F = pd.DataFrame([F])
# # # print(F.iloc[0])
# #
# # # parameters of feed gas
# # F0 = {}
# P0, T0 = feed_dict["condition"]["P"], feed_dict["condition"]["T"]  # P0 bar, T0 K
# v0 = feed_dict["condition"]["Sv"]   # space velocity per tube, m3/s
# # # print("v0,%s" % v0)
# Ft0 = P0 * 1e5 * v0 / R / T0  # mol/s
# #
# feed_info = {"T": T0, "P": P0, "Sv": v0, "Ft0": Ft0}
# print(rate(483, 30, F.iloc[0], feed_info, chem_dict))
#
