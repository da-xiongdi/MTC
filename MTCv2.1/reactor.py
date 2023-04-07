import json
import numpy as np
import scipy
from CoolProp.CoolProp import PropsSI
import pandas as pd
from matplotlib import pyplot as plt

R = 8.314  # J/mol/K
ks, vof = 0.2, 0.8


class Reaction:
    """
    simulation of adiabatic reactor for conversion of CO2 to CH3OH
    """

    def __init__(self, kn_model):
        self.kn_model = kn_model
        if kn_model == 'GR':
            chem_path = 'in_chem_GR.json'
        elif kn_model == 'BU':
            chem_path = 'in_chem_BU_revised.json'
        elif kn_model == 'SL':
            chem_path = 'in_chem_SL.json'

        in_path = {'chem': chem_path, 'reactor': 'in_reactor.json', 'feed': 'in_feed.json'}
        in_data = dict()
        for key, values in in_path.items():
            with open(values) as f:
                in_data[key] = json.load(f)

        # reactor parameters
        self.react_para = in_data['reactor']["reactor"]
        self.L, self.Dt = in_data['reactor']["reactor"]['L'], in_data['reactor']["reactor"]['Dt']  # length, m
        self.nrt = in_data['reactor']["reactor"]['nrt']  # number of the reaction tube
        self.phi = in_data['reactor']["reactor"]["phi"]  # void of fraction
        self.rhoc = in_data['reactor']["reactor"]["rhoc"]  # density of catalyst, kg/m3
        self.insulator_para = in_data['reactor']["insulator"]

        # prescribed chem data of reaction
        self.chem_data = in_data['chem']
        self.comp_list = in_data['chem']["comp_list"]
        self.react_num = len(in_data['chem']["kr"])
        self.react_sto = np.empty((self.react_num, 5))
        # self.react_dH = np.empty(self.react_num)
        for i in range(self.react_num):
            key = str(i + 1)
            self.react_sto[i] = in_data['chem']["stoichiometry"][key]
            # self.react_dH[i] = in_data['chem']["heat_reaction"][key]

        # feed gas parameter
        self.feed_para = in_data['feed']["condition"]
        self.F0 = np.zeros(len(self.comp_list))  # component of feed gas, mol/s; ndarray
        self.P0, self.T0 = in_data['feed']["condition"]["P"], in_data['feed']["condition"]["T"]  # P0 bar, T0 K
        # volumetric flux per tube from space velocity
        self.sv = in_data['feed']["condition"]["Sv"]
        self.v0 = self.sv * self.L * np.pi * self.Dt ** 2 / 4 / 3600 / self.nrt  # volumetric flux per tube, m3/s

        self.Ft0 = self.P0 * 1e5 * self.v0 / R / self.T0  # total flux of feed,mol/s
        if in_data['feed']["condition"]["recycle"] == "off":  # fresh stream
            self.F0[0] = self.Ft0 / (in_data['feed']["condition"]["H2/CO2"] + 1)
            self.F0[1] = self.Ft0 - self.F0[0]
        elif in_data['feed']["condition"]["recycle"] == "on":  # recycled stream
            self.F0 = np.array([float(i) for i in in_data['feed']["feed"].split('\t')])

    @staticmethod
    def react_H(T, in_dict):
        dH = np.zeros(len(in_dict["heat_reaction"].keys()))
        i = 0
        for key, value in in_dict["heat_reaction"].items():
            dH[i] = -(value[0] * T + value[1]) * 1e-6
            i += 1
        # print(dH)
        return dH

    @staticmethod
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

    @staticmethod
    def keq(T, in_dict):
        """
        calculate the equilibrium constant
        :param T: operating temperature
        :param in_dict: prescribed chemical parameter
        :return: equilibrium constant
        """
        react_eq_constant = dict()
        for key, value in in_dict["keq"].items():
            react_eq_constant[key] = 10 ** (value[0] / T + value[1])
        return react_eq_constant

    @staticmethod
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

    def rate_bu(self, T, Pi):
        """
        calculate the reaction rate
        :param T: operating temperature, K
        :param Pi: partial pressure of each component, bar
        :return: reaction rate of each component for each and all reaction; mol/s/kg_cat
        """
        # convert the partial pressure from ndarray to pd.Series
        Pi = pd.Series(Pi, index=self.comp_list)

        # calculate the reaction constant
        rate_const = self.kr(T, self.chem_data)
        ad_const = self.kad(T, self.chem_data)
        eq_const = self.keq(T, self.chem_data)

        # calculate the rate of each reaction
        react_rate = np.zeros(self.react_num)
        driving = rate_const['1'] * Pi['CO2'] * Pi['H2'] * (
                1 - Pi['H2O'] * Pi["Methanol"] / Pi["H2"] ** 3 / Pi['CO2'] / eq_const['1'])
        inhibiting = (1 + ad_const["H2O/H2"] * Pi['H2O'] / Pi['H2'] +
                      ad_const["H2"] * Pi["H2"] ** 0.5 + ad_const["H2O"] * Pi["H2O"])
        react_rate[0] = driving / inhibiting ** 3

        driving = rate_const['2'] * Pi['CO2'] * (1 - Pi['H2O'] * Pi["CO"] / Pi["H2"] / Pi['CO2'] / eq_const['2'])
        react_rate[1] = driving / inhibiting

        # compute the reaction rate for each component in every reaction
        react_comp_rate = self.react_sto * np.repeat(react_rate, 5).reshape(self.react_num, 5)
        react_comp_rate = np.vstack((react_comp_rate, np.sum(react_comp_rate, axis=0).T))
        # react_comp_rate = np.hstack((react_comp_rate, np.array([0, 0, 0]).reshape(3, 1)))

        return react_comp_rate

    def rate_sl(self, T, Pi):
        """
        calculate the reaction rate
        :param T: operating temperature, K
        :param Pi: partial pressure of each component, bar
        :return: reaction rate of each component for each and all reaction; mol/s/kg_cat
        """
        # convert the partial pressure from ndarray to pd.Series
        Pi = pd.Series(Pi, index=self.comp_list)

        # calculate the reaction constant
        rate_const = self.kr(T, self.chem_data)
        # print(rate_const)
        ad_const = self.kad(T, self.chem_data)
        eq_const = self.keq(T, self.chem_data)

        # calculate the rate of each reaction
        react_rate = np.zeros(self.react_num)
        driving = rate_const['1'] * Pi['CO2'] * Pi['H2']**2 * (
                1 - Pi['H2O'] * Pi["Methanol"] / Pi["H2"] ** 3 / Pi['CO2'] / eq_const['1'])
        inhibiting = (ad_const["H2"] * Pi['H2']**0.5 +
                      ad_const["H2O"] * Pi["H2O"] + Pi["Methanol"])
        react_rate[0] = driving / inhibiting ** 2

        driving = rate_const['2'] * Pi['CO2'] * (1 - Pi['H2O'] * Pi["CO"] / Pi["H2"] / Pi['CO2'] / eq_const['2'])
        react_rate[1] = driving / inhibiting

        # compute the reaction rate for each component in every reaction
        react_comp_rate = self.react_sto * np.repeat(react_rate, 5).reshape(self.react_num, 5)
        react_comp_rate = np.vstack((react_comp_rate, np.sum(react_comp_rate, axis=0).T))
        # react_comp_rate = np.hstack((react_comp_rate, np.array([0, 0, 0]).reshape(3, 1)))

        return react_comp_rate

    def rate_gr(self, T, Pi):
        """
        calculate the reaction rate
        :param T: operating temperature, K
        :param Pi: partial pressure of each component, bar
        :return: reaction rate of each component for each and all reaction; mol/s/kg_cat
        """

        # convert the partial pressure from ndarray to pd.Series
        Pi = pd.Series(Pi, index=self.comp_list)

        # calculate the reaction constant
        rate_const = self.kr(T, self.chem_data)
        ad_const = self.kad(T, self.chem_data)
        eq_const = self.keq(T, self.chem_data)

        # calculate the rate of each reaction
        react_rate = np.zeros(self.react_num)
        driving = rate_const['1'] * ad_const['CO2'] * (
                Pi['CO2'] * Pi['H2'] ** 1.5 - Pi['H2O'] * Pi["Methanol"] / Pi["H2"] ** 1.5 / eq_const['1'])
        inhibiting = (1 + ad_const["CO"] * Pi['CO'] + ad_const["CO2"] * Pi['CO2']) * \
                     (Pi["H2"] ** 0.5 + ad_const["H2O/H2"] * Pi["H2O"])
        react_rate[0] = driving / inhibiting

        driving = rate_const['2'] * ad_const['CO2'] * (Pi['CO2'] * Pi["H2"] - Pi['H2O'] * Pi["CO"] / eq_const['2'])
        react_rate[1] = driving / inhibiting

        driving = rate_const['3'] * ad_const['CO'] * (
                Pi['CO'] * Pi["H2"] ** 1.5 - Pi['Methanol'] / Pi["H2"] ** 0.5 / eq_const['3'])
        react_rate[2] = driving / inhibiting

        # compute the reaction rate for each component in every reaction
        react_comp_rate = self.react_sto * np.repeat(react_rate, 5).reshape(self.react_num, 5)
        react_comp_rate = np.vstack((react_comp_rate, np.sum(react_comp_rate, axis=0).T))
        # react_comp_rate = np.hstack((react_comp_rate, np.array([0, 0, 0, 0]).reshape(4, 1)))

        return react_comp_rate

    def balance(self, T, P, F_dict):
        """
        energy and material balance in the reactor
        :param T: operating temperature, K
        :param P: operating pressure, bar
        :param F_dict: molar flow rate of each component, mol/s; ndarray
        :return: temperature and molar flux variation of gas
        """
        Ft = np.sum(F_dict)  # total molar flow rate

        # calculate the partial pressure
        # calculate the correction to volumetric flow rate (m3/s)
        v = self.v0 * (self.P0 / P) * (T / self.T0) * (Ft / self.Ft0)
        Pi = F_dict * R * T / v * 1e-5  # bar

        # calculate the change of the molar flow rate due to reactions, mol/s/kg_cat
        if self.kn_model == 'GR':
            dF_react = self.rate_gr(T, Pi)
        elif self.kn_model == 'BU':
            dF_react = self.rate_bu(T, Pi)
        elif self.kn_model == 'SL':
            dF_react = self.rate_sl(T, Pi)

        # calculate the change of enthalpy due to reaction, kJ/(kg_cat s)
        dH_react = self.react_H(T, self.chem_data)
        if self.react_num == 3: dH_react[2] = dH_react[0] - dH_react[1]
        dH = np.matmul(dF_react[:-1, 0], dH_react.T)

        # calculate the heat capacity of each component, cp*n, J/(s K)
        heat_capacity = 0
        for i in range(5):
            # read the heat capacity for each component, J/(mol K)
            cp = PropsSI('CPMOLAR', 'T', T, 'P', Pi[i] * 1e5, self.comp_list[i]) if Pi[i] > 0 else 0
            heat_capacity += cp * F_dict[i]
        dT = dH * 1e3 / heat_capacity  # K/kg_cat

        return dF_react[-1], dT

# a = [343, 0.012360681152337636, 0.0039033729954750422, 0.03]
# b = [524.5764062395502, 0.012489109102363472, 0.030998751112742237, 0.015]
# c = [63.685662602388845, 35.503136000940216, 4.5e-05, 1.4e-05, 0.2930081289718913]
# ysol = reactor.ode_multi(b, a, 50, c, 0.6)
#
# xsol = np.linspace(0.015, 0.03, 200)
# # ysol = res.sol(xsol)
# fig, axe = plt.subplots(2, 2)
# # [xc, xd, Nd, T, dTdz]
# axe[0][0].plot(xsol, ysol[0])
# axe[0][0].plot(xsol, ysol[1])
# axe[0][0].legend(["CH3OH", "H2O"])
# axe[0][1].plot(xsol, ysol[2])
# axe[1][0].plot(xsol, ysol[3])
# axe[1][1].plot(xsol, ysol[4])
# plt.show()
#
# print(ysol[1][0],ysol[1][-1])
# print(ysol[1][-1] - b[2])
# print(ysol[1][0] - a[2])
# c1 = [550.9956251927392, 0.006494309709190364, 0.016514917211696244, 0.035]
# c2 = [353, 0, 0, 0.015]
# c3 = [58.02352864647421, 35.9547344923201, 4.5e-05, 1.4e-05, 0.31750891539283604]
# r = 0.14


# gap = 1e5
# for r in np.arange(0, 2, 0.01):
#     res = ode_multi(c2, c1, 50, c3, r)
#     temp = abs(res[1][-1] - c1[2])
#     if temp < gap:
#         r_sel = r
#         gap = temp
#
# print(r_sel, gap)
# ode_multi(c2, c1, 50, c3, r=1.25)

# T = 93+273
# print(PropsSI('P', 'T', T, 'Q', 1, 'Ammonia'))
# mix_liquid = 'HEOS::H2O[0.9]&Ammonia[0.1]'#'HEOS::NH3[%s]&H2O[%s]' % (0.1, 0.9)
# Pl_sat = PropsSI('P', 'T', T, 'Q', 1, mix_liquid)
# print(Pl_sat)

# for i in np.arange(350,380):
#     a.append(PropsSI('P', 'T', i, 'Q', 1, 'Methanol'))
#     b.append(PropsSI('P', 'T', i, 'Q', 1, 'H2O'))
#
# plt.plot(np.arange(350,380),a)
# plt.plot(np.arange(350,380),b)
# plt.show()
