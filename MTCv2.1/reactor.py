import numpy as np
from CoolProp.CoolProp import PropsSI
import pandas as pd

R = 8.314  # J/mol/K
ks, vof = 0.2, 0.8


class Reaction:
    """
    basic simulation of CO2 to CH3OH
    energy and mass balance are calculated
    """

    def __init__(self, reactor_para, chem_para, feed_para):

        # reactor parameters
        self.react_para = reactor_para
        self.L, self.Dt = self.react_para['L'], self.react_para['Dt']  # length, m
        self.nrt = self.react_para['nrt']  # number of the reaction tube
        self.phi = self.react_para["phi"]  # void of fraction
        self.rhoc = self.react_para["rhoc"]  # density of catalyst, kg/m3

        # prescribed chem data of reaction
        self.comp_list = ["CO2", "H2", "Methanol", "H2O", "CO"]
        self.chem_data = chem_para
        self.react_num = len(self.chem_data["kr"])
        self.react_sto = np.empty((self.react_num, 5))
        self.kn_model = self.chem_data['kn_model']
        for i in range(self.react_num):
            key = str(i + 1)
            self.react_sto[i] = self.chem_data["stoichiometry"][key]

        # feed gas parameter
        self.feed_para = feed_para
        self.P0, self.T0 = self.feed_para["P"], self.feed_para["T"]  # P0 bar, T0 K

        if self.feed_para["recycle"] == 1:  # fresh stream
            self.F0 = np.zeros(len(self.comp_list))  # component of feed gas, mol/s; ndarray
            # volumetric flux per tube from space velocity
            self.sv = self.feed_para["Sv"]
            self.v0 = self.sv * self.L * np.pi * self.Dt ** 2 / 4 / 3600 / self.nrt  # volumetric flux per tube, m3/s
            self.Ft0 = self.P0 * 1e5 * self.v0 / R / self.T0  # total flux of feed,mol/s

            self.F0[0] = 1 / (1 + 1 * self.feed_para["H2/CO2"] + self.feed_para['CO/CO2']) * self.Ft0
            self.F0[4] = self.F0[0] * self.feed_para['CO/CO2']
            self.F0[1] = self.Ft0 - self.F0[0] - self.F0[4]
        else:  # recycled stream
            self.F0 = self.feed_para[self.comp_list].to_numpy()
            self.Ft0 = np.sum(self.F0)
            self.v0 = self.Ft0 * R * self.T0 / (self.P0 * 1e5)
            self.sv = self.v0 * self.nrt*3600*4/self.L/np.pi/self.Dt**2

    @staticmethod
    def react_H(T, in_dict):
        dH = np.zeros(len(in_dict["heat_reaction"].keys()))
        i = 0
        for key, value in in_dict["heat_reaction"].items():
            dH[i] = -(value[0] * T + value[1]) * 1e-6
            i += 1
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
        driving = rate_const['1'] * Pi['CO2'] * Pi['H2'] ** 2 * (
                1 - Pi['H2O'] * Pi["Methanol"] / Pi["H2"] ** 3 / Pi['CO2'] / eq_const['1'])
        inhibiting = (ad_const["H2"] * Pi['H2'] ** 0.5 +
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


