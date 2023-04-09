import json
import os.path
import sys

import numpy as np

R = 8.314


class ReadData:
    """
    read data for simulation of conversion from CO2 to CH3OH
    """

    def __init__(self, kn_model='BU', in_path=None):

        # chem data of reaction for kinetic model
        if kn_model == 'BU':
            self.chem = self.kn_bu
        elif kn_model == 'SL':
            self.chem = self.kn_sl
        elif kn_model == 'GR':
            self.chem = self.kn_gr
        else:
            print('kn_model should be one of BU, GR, and SL')
            sys.exit(1)
        self.comp_list = ["CO2", "H2", "Methanol", "H2O", "CO"]
        self.react_num = len(self.chem["kr"])
        self.react_sto = np.empty((self.react_num, 5))
        for i in range(self.react_num):
            key = str(i + 1)
            self.react_sto[i] = self.chem["stoichiometry"][key]

        # data path for reactor and feed gas
        self.root_path = sys.path[0]
        if in_path is None:
            in_path = {'reactor': 'in_reactor.json', 'feed': 'in_feed.json'}
        in_data = dict()
        for key, values in in_path.items():
            try:
                file_path = os.path.join(self.root_path, values)
                with open(file_path) as f:
                    in_data[key] = json.load(f)
            except FileNotFoundError:
                with open(values) as f:
                    in_data[key] = json.load(f)

        # reactor parameters
        self.react_para = in_data['reactor']["reactor"]
        self.L, self.Dt = self.react_para['L'], self.react_para['Dt']  # length, m
        self.nrt = self.react_para['nrt']  # number of the reaction tube
        self.phi = self.react_para["phi"]  # void of fraction
        self.rhoc = self.react_para["rhoc"]  # density of catalyst, kg/m3
        self.insulator_para = in_data['reactor']["insulator"]

        # feed gas parameter
        self.feed_para = in_data['feed']["condition"]
        self.F0 = np.zeros(len(self.comp_list))  # component of feed gas, mol/s
        self.P0, self.T0 = self.feed_para["P"], self.feed_para["T"]  # P0 bar, T0 K
        self.sv = self.feed_para["Sv"]  # volumetric flux per tube from space velocity

    @property
    def hr(self):
        """
        parameter for the calculation of reaction enthalpy, dH = aT^4+b
        ref: Cui, 2020, Chemical Engineering Journal, 10.1016/j.cej.2020.124632
        :return: [a, b] for reactions
        """
        heat_reaction = {
            "1": [3.589e4, 4.0047e7],
            "2": [9.177e3, -4.4325e7]
        }
        return heat_reaction

    @property
    def keq(self):
        """
        parameter for the calculation of equilibrium constant, k = 10^(a/T+b)
        ref: Graaf, 1986, Chemical Engineering Science, 10.1016/0009-2509(86)80019-7
        :return: [a, b] for reactions
        """
        keq = {
            "1": [3066, -10.92],
            "2": [-2073, 2.029]
        }
        return keq

    @property
    def kn_sl(self):
        """
        reaction kinetic model proposed by Slotboom
        ref: Slotboom, 2020, Chemical Engineering Journal, 10.1016/j.cej.2020.124181
        :return: parameter dict
        """
        stoichiometry = {
            "1": [-1, -3, 1, 1, 0],
            "2": [-1, -1, 0, 1, 1]
        }
        kad = {
            "H2": [1.099, 0],
            "H2O": [126.4, 0]
        }
        kr = {
            "1": [7.414e14, -166000],
            "2": [1.111e19, -203700]
        }
        chem_data = {"stoichiometry": stoichiometry, "kad": kad, "kr": kr, "keq": self.keq, 'heat_reaction': self.hr}
        return chem_data

    @property
    def kn_bu(self):
        """
        reaction kinetic model proposed by Bussche
        ref: Bussche, 1996, Journal of Catalysis, 10.1006/jcat.1996.0156
        :return: parameter dict
        """
        stoichiometry = {
            "1": [-1, -3, 1, 1, 0],
            "2": [-1, -1, 0, 1, 1]
        }
        kad = {
            "H2": [0.499, 17197],
            "H2O": [6.62e-11, 124119],
            "H2O/H2": [3453.38, 0]
        }
        kr = {
            "1": [1.07, 40000],
            "2": [1.22e10, -98084]
        }
        chem_data = {"stoichiometry": stoichiometry, "kad": kad, "kr": kr, "keq": self.keq, 'heat_reaction': self.hr}
        return chem_data

    @property
    def kn_gr(self):
        """
        reaction kinetic model proposed by graaf
        ref: Graaf, 1988, Chemical Engineering Science, 10.1016/0009-2509(88)85127-3
        :return: parameter dict
        """
        stoichiometry = {
            "1": [-1, -3, 1, 1, 0],
            "2": [-1, -1, 0, 1, 1],
            "3": [0, -2, 1, 0, -1]
        }
        kad = {
            "CO": [7.99e-7, 58100],
            "CO2": [1.02e-7, 67400],
            "H2O/H2": [4.13e-11, 104500]
        }
        kr = {
            "1": [436, -65200],
            "2": [7.31e8, -123400],
            "3": [2.69e7, -109900]
        }
        chem_data = {"stoichiometry": stoichiometry, "kad": kad, "kr": kr, "keq": self.keq, 'heat_reaction': self.hr}
        return chem_data


data = ReadData(kn_model='BU')
print(data.chem)

# print(data.kn_bu["kr"])
# react_rate_constant = {}
#
# for key, value in data.kn_bu["kr"].items():
#     react_rate_constant[key] = value[0] * np.exp(value[1] / 503 / R)
#
# print(react_rate_constant)
