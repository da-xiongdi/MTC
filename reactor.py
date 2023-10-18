import numpy as np
from CoolProp.CoolProp import PropsSI
import pandas as pd

from prop_calculator import VLE, mixture_property

R = 8.314  # J/mol/K
ks, vof = 0.2, 0.8  # 0.2 1.5 for 0.42 1 for 0.3 0.2 for 0.15 # 1, 0.4 for CO exp


class Reaction:
    """
    basic simulation of CO2 to CH3OH
    energy and mass balance are calculated
    """

    def __init__(self, L, D, n, phi, rho, chem_para, T0, P0, F0, eos):

        # 0 for ideal 1 for SRK
        self.eos = eos

        # reactor parameters
        self.L, self.Dt, self.n = L, D, n
        self.phi, self.rho = phi, rho
        self.ds = 5e-3  # catalyst particle diameter

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
        self.R = 8.314
        self.P0, self.T0 = P0, T0  # P0 bar, T0 K
        self.F0, self.Ft0 = F0, np.sum(F0)
        self.v0 = self.Ft0 * self.R * self.T0 / (self.P0 * 1e5)

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

    def ergun(self, T, P, F_dict):
        """
        energy and material balance in the reactor
        :param T: operating temperature, K
        :param P: operating pressure, bar
        :param F_dict: molar flow rate of each component, mol/s; ndarray
        :return: pressure drop per length, pa/m
        """
        Ft = np.sum(F_dict)
        v = self.v0 * (self.P0 / P) * (T / self.T0) * (Ft / self.Ft0)
        u = v / (np.pi * self.Dt ** 2 / 4)
        if self.eos == 1:
            properties = VLE(T, comp=self.comp_list)
            _, z = properties.phi(pd.Series(F_dict / Ft, index=self.comp_list), P)
        elif self.eos == 0:
            z = 1
        gas_property = mixture_property(T, pd.Series(F_dict / Ft, index=self.comp_list), z, rho_only=False)
        Re = self.ds * u * gas_property['rho'] / gas_property['vis'] / self.phi
        drop_per_length = - (150 / Re + 1.75) * self.phi / (1 - self.phi) ** 3 * (
                gas_property['rho'] * u ** 2 / self.ds)  # Pa/m
        return drop_per_length

    def convection(self, T, P, F_dict):
        """
        :param T: temperature of reactor gas, K
        :param P: pressure of reactor, bar
        :param F_dict: molar flow rate of each component, mol/s; ndarray
        :return: convection heat transfer coefficient, W/m2 K
        """
        Ft = np.sum(F_dict)
        xi = F_dict / Ft * P
        mix_property = mixture_property(T, pd.Series(xi, self.comp_list), Pt=P)
        M = 0.25 * 44 + 0.75 * 2
        Pr = mix_property['vis'] * (mix_property['cp_m'] / (M / 1000)) / mix_property['k']
        v = self.v0 * (self.P0 / P) * (T / self.T0) * (Ft / self.Ft0)  # m3/s
        u = v / (np.pi * self.Dt ** 2 / 4)
        Re = u * self.Dt * mix_property['rho'] / mix_property['vis']
        if Re > 1e4:
            Nu = 0.0265 * Re ** 0.8 * Pr ** 0.3
        elif 2300 < Re < 1e4:
            f = (0.79 * np.log(Re) - 1.64) ** -2
            Nu = f / 8 * (Re - 1000) * Pr / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
        elif Re < 2300:
            Nu = 3.66
        h = Nu * mix_property['k'] / self.Dt  # W/m K
        return h

    def rate_vi(self, T, Pi):
        """
        calculate the reaction rate
        :param T: operating temperature, K
        :param Pi: partial pressure of each component, bar
        :return: reaction rate of each component for each and all reaction; mol/s/kg_cat
        """

        # convert the partial pressure from ndarray to pd.Series
        Pi = pd.Series(Pi, index=self.comp_list)
        K_H2O = 96808*np.exp(-51979/8.314/T)
        k_r = 11101.2*np.exp(-117432/8.314/T)
        Ke = 1/np.exp(-12.11+5319/T+1.012*np.log(T)+1.144*10**(-4*T))
        react_rate = k_r*(Pi["CO2"]-Pi["CO"]*Pi["H2O"]/Ke/Pi["H2"])/(1+K_H2O*Pi["H2O"]/Pi["H2"])*1000

        react_comp_rate = np.zeros((3, 5))
        react_comp_rate[1] = react_rate * self.react_sto[1]
        react_comp_rate[2] = react_rate * self.react_sto[1]
        # print(react_comp_rate)
        return react_comp_rate

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
        xi = F_dict / Ft  # molar fraction of mix

        # calculate the partial pressure/fugacity
        # calculate the correction to volumetric flow rate (m3/s)
        if self.eos == 1:
            # fugacity coe, compression factor
            vle_cal = VLE(T, self.comp_list)
            phi, _ = vle_cal.phi(comp=pd.Series(xi, index=self.comp_list), P=P, phase=0)
        else:
            phi = 1
        v = self.v0 * (self.P0 / P) * (T / self.T0) * (Ft / self.Ft0)
        Pi = F_dict * R * T / v * 1e-5  # bar
        fi = Pi * phi
        # calculate the change of the molar flow rate due to reactions, mol/s/kg_cat
        if self.kn_model == 'GR':
            dF_react = self.rate_gr(T, fi)
        elif self.kn_model == 'BU':
            dF_react = self.rate_bu(T, fi)  # self.rate_vi(T, fi) #
        elif self.kn_model == 'SL':
            dF_react = self.rate_sl(T, fi)
        elif self.kn_model == "VI":
            dF_react = self.rate_vi(T, fi)

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
        res = {
            'mflux': dF_react[-1],
            'tc': heat_capacity,
            'hflux': dH * 1e3,
            'Tvar': dT
        }
        return res

    # F = np.array([0.062228879, 0.198288202, 0.009296752, 0.012506192, 0.012891983])
# comp = ['CO2', 'H2', 'Methanol', 'H2O', 'CO']
# T = 529
# P = 70
#
# from read import ReadData
#
# # prepare data for the simulation
# in_data = ReadData(kn_model='BU')
# reactor_data = in_data.reactor_data
# feed_data = in_data.feed_data
# chem_data = in_data.chem
# insulator_data = in_data.insulator_data
#
# for i in range(feed_data.shape[0]):
#     for j in range(reactor_data.shape[0]):
#         for k in range(insulator_data.shape[0]):
#             insulator_data['Din'].iloc[k] = reactor_data['Dt'].iloc[j]
#
#             a = Reaction(reactor_data.iloc[j], chem_data, feed_data.iloc[i])
#             a.convection(T, P, F, comp)
