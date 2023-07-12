import json
import os.path
import sys

import numpy as np
import pandas as pd

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
        self.chem['kn_model'] = kn_model

        # data path for reactor and feed gas
        self.root_path = sys.path[0]
        if in_path is None:
            in_path = {'reactor': 'in_reactor.json', 'feed': 'in_feed.json', 'insulator': "in_insulator.json"}
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
        self.react_para = in_data["reactor"]["reactor"]

        # insulator parameters
        self.insulator_para = in_data["insulator"]["insulator"]

        # feed gas parameter
        self.feed_para = in_data['feed']

    @property
    def feed_data(self):

        if self.feed_para["condition"]["fresh"] == 'on':
            feed = self.feed_para["condition"]
            H2_CO2 = feed["H2/CO2"]
            recycle = 1

            # feed para frame
            T0_array = self.data_array(feed["T"])
            T_feed_array = self.data_array(feed["T_feed"])
            P0_array = self.data_array(feed["P"])
            sv_array = self.data_array(feed["Sv"])
            H2_array = self.data_array(feed['H2'])
            CO_CO2_array = self.data_array(feed['CO/CO2'])
            feed_num = len(T0_array) * len(P0_array) * len(sv_array) * len(CO_CO2_array)
            feed_para = pd.DataFrame(index=np.arange(feed_num), columns=list(feed.keys()))
            i = 0
            for T in T0_array:
                for P in P0_array:
                    for sv in sv_array:
                        for CO_CO2 in CO_CO2_array:
                            for H2 in H2_array:
                                for T_feed in T_feed_array:
                                    feed_para.iloc[i] = np.array([T, T_feed, P, recycle, H2_CO2, CO_CO2, H2, sv])
                                    i += 1
        else:
            feed = np.array([float(i) for i in self.feed_para["feed"].split('\t')])
            T = self.feed_para["condition"]['T'][0]
            T_feed = self.feed_para["condition"]['T_feed'][0]
            P = self.feed_para["condition"]['P'][0]
            feed = np.append(np.array([T, T_feed,P, 0]), feed)
            feed_para = pd.DataFrame(feed.reshape(1, len(feed)),
                                     columns=['T', 'T_feed', 'P', 'fresh', "CO2", "H2", "Methanol", "H2O", "CO"])
        return feed_para

    @property
    def insulator_data(self):
        # insulator parameters

        status = 1 if self.insulator_para['status'] == 'on' else 0
        location = 0 if self.insulator_para["io"] == 'in' else 1
        nit = self.insulator_para["nit"]  # tube number of the insulator
        Thick = self.insulator_para['Thick']
        qm = self.insulator_para['qm']
        q = self.insulator_para['q']

        # insulator para frame
        Din_array = self.data_array(self.insulator_para['Din'])  # Din should be same with the Dt of reactor
        Tc_array = self.data_array(self.insulator_para['Tc'])

        insulator_num = len(Tc_array)
        insulator_para = pd.DataFrame(index=np.arange(insulator_num), columns=list(self.insulator_para.keys()))
        i = 0
        for Din in Din_array:
            for Tc in Tc_array:
                insulator_para.iloc[i] = [status, Din, Thick, nit, Tc, location, qm, q]
                i += 1

        return insulator_para

    @property
    def reactor_data(self):
        L1 = self.react_para['L1']  # length, m
        nrt = self.react_para['nrt']  # number of the reaction tube
        phi = self.react_para["phi"]  # void of fraction
        rhoc = self.react_para["rhoc"]  # density of catalyst, kg/m3
        stage = self.react_para['stage']
        L2 = self.react_para['L2']  # length, m
        Uc = self.react_para["Uc"]  # total heat transfer coefficient of the reactor, W/m2 K
        recycle = 1 if self.react_para['recycle'] == "on" else 0

        # reactor para frame
        Dt_array = self.data_array(self.react_para['Dt'])
        reactor_num = len(Dt_array)
        react_para = pd.DataFrame(index=np.arange(reactor_num), columns=list(self.react_para.keys()))
        i = 0
        for Dt in Dt_array:
            react_para.iloc[i] = [L1, Dt, nrt, rhoc, phi, recycle, stage, L2, Uc]
            i += 1
        return react_para

    @staticmethod
    def data_array(in_data):
        try:
            data_lenth = len(in_data)
        except TypeError:
            print("Value in json should be list!")
            sys.exit(1)
        if data_lenth != 3:
            out_data = np.array(in_data)
        elif data_lenth == 3:
            out_data = np.linspace(in_data[0], in_data[1], in_data[2])
        return out_data

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

# data = ReadData(kn_model='BU')
# # print(data.feed_para.keys())
# # print(pd.DataFrame(columns=data.feed_para.keys()))
# print(data.feed_data)
# print(data.insulator_data)
# print(data.reactor_data)
# print()
# print(data.kn_bu["kr"])
# react_rate_constant = {}
#
# for key, value in data.kn_bu["kr"].items():
#     react_rate_constant[key] = value[0] * np.exp(value[1] / 503 / R)
#
# print(react_rate_constant)
