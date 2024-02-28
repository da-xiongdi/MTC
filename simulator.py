import os.path
import warnings

import chemicals
import scipy
from fluids.numerics import OscillationError

from prop_calculator import mixture_property, VLEThermo
from insulator import Insulation, ks
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from datetime import datetime
from reactor import Reaction

# from gibbs import Gibbs
RED = '\033[91m'
ENDC = '\033[0m'


class Simulation:
    def __init__(self, reactors_para, chem_para, feed_para, insulator_para, eos, drop):

        # basic info
        self.comp_list = ["CO2", "H2", "Methanol", "H2O", "CO", 'Ar']
        self.R = 8.314
        self.eos = eos  # 0 for ideal 1 for SRK
        self.drop = drop  # 1 for ergun 0 for zero drop

        # reactor insulator feed para
        self.reactors_para = reactors_para  #

        self.reactor_para = self._reactor_para()  # rewrite the reactor para
        self.chem_para = chem_para
        self.chem_para = chem_para
        self.feed_para = feed_para
        self.feed0_para = self._feed_para()
        F_in = self.feed0_para.loc['F0']
        self.comp_num = len(F_in)
        if len(F_in) == 6 and F_in[-1] == 0:
            self.F0 = self.F0[:-1]
            self.feed0_para.loc['F0'] = F_in[:-1]
            self.comp_num -= 1
            self.comp_list = self.comp_list[:-1]
        self.insulators = insulator_para
        self.insulator = self._insulator_para()

    def _reactor_para(self):
        """
        generate para for each reactor in series (pd.Dataframe)
        [[stage1, paras...], [stage2, paras...]]
        """
        self.stage = self.reactors_para['stage']
        self.Uc = self.reactors_para['Uc']
        self.recycle = self.reactors_para['recycle']
        Dt_name = [f'Dt{n + 1}' for n in range(self.stage)]
        L_name = [f'L{n + 1}' for n in range(self.stage)]
        Din_name = [f'Dc{n + 1}' for n in range(self.stage)]
        self.L, self.Dt = self.reactors_para[L_name], self.reactors_para[Dt_name]
        self.Din = self.reactors_para[Din_name]
        self.nrt = self.reactors_para['nrt']
        reactors = pd.DataFrame(index=np.arange(self.stage),
                                columns=['L', 'Dt', 'Dc'] + self.reactors_para.index[self.stage * 3:].tolist())
        for n in range(self.stage):
            reactors.loc[n, ['L', 'Dt', 'Dc']] = [self.L[n], self.Dt[n], self.Din[n]]
            reactors.iloc[n, 3:] = self.reactors_para[3 * self.stage:]
        return reactors

    def _feed_para(self):
        """
        generate feed para
        """
        self.P0, self.T0 = self.feed_para["P"], self.feed_para["T"]  # P0 bar, T0 K
        self.T_feed = self.feed_para["T_feed"]
        self.vle_cal = VLEThermo(self.comp_list)
        if self.feed_para["fresh"] == 1:  # the feed to the plant is fresh stream
            self.F0 = np.zeros(len(self.comp_list))  # component of feed gas, mol/s; ndarray
            # volumetric flux per tube from space velocity
            if self.feed_para["H2"] == 0:
                self.sv = self.feed_para["Sv"]
                # volumetric flux per tube under input temperature and pressure, m3/s
                self.v0 = self.sv * self.L[0] * np.pi * self.Dt[0] ** 2 / 4 / 3600 / self.nrt
                self.Ft0 = self.P0 * 1e5 * self.v0 / self.R / self.T0  # total flux of feed,mol/s
                self.F0[0] = 1 / (1 + 1 * self.feed_para["H2/CO2"] + self.feed_para['CO/CO2']) * self.Ft0
                self.F0[4] = self.F0[0] * self.feed_para['CO/CO2']
                self.F0[1] = self.Ft0 - self.F0[0] - self.F0[4]
                x0 = self.F0 / self.Ft0
                z0 = self.vle_cal.z(T=self.T0, P=self.P0, x=self.F0)
                self.Ft0 = self.Ft0 / z0
                self.F0 = x0 * self.Ft0
                self.H2 = self.F0[1] * 8.314 * 273.15 / 1e5 * 3600  # Nm3/h
            else:
                self.H2 = self.feed_para["H2"]  # Nm3/h
                self.F0[1] = self.H2 / 3600 * 1e5 / self.R / 273.15  # mol/s
                self.F0[0] = self.F0[1] / self.feed_para["H2/CO2"]
                self.F0[4] = self.F0[0] * self.feed_para['CO/CO2']
                self.Ft0 = np.sum(self.F0)
                z0 = self.vle_cal.z(T=self.T0, P=self.P0, x=self.F0)
                self.v0 = self.Ft0 * self.R * self.T0 * z0 / (self.P0 * 1e5)
                self.sv = self.v0 * self.nrt * 3600 * 4 / self.L[0] / np.pi / self.Dt[0] ** 2
            # print(self.sv, self.H2)
        else:  # recycled stream
            self.F0 = self.feed_para[self.comp_list].to_numpy()
            self.Ft0 = np.sum(self.F0)
            z0 = self.vle_cal.z(T=self.T0, P=self.P0, x=self.F0)
            self.v0 = self.Ft0 * self.R * self.T0 * z0 / (self.P0 * 1e5)
            self.sv = self.v0 * self.nrt * 3600 * 4 / self.L[0] / np.pi / self.Dt[0] ** 2
            self.H2 = self.F0[1] * self.R * 273.15 / 1E5
        feed = pd.Series([self.T0, self.P0, self.F0], index=['T0', 'P0', 'F0'])
        return feed

    def _insulator_para(self):
        """
        reconstruct insulator para
        :return: [[stage1, paras...], [stage2, paras...]] pd.Dataframe
        """
        paras_stage = ['Din', 'Thick', 'Tc', 'qmc', 'Th', 'qmh']
        paras_array_name = {'status_name': [f'status{n + 1}' for n in range(self.stage)],
                            'pattern_name': [f'pattern{n + 1}' for n in range(self.stage)]}
        for n in range(self.stage):
            for para_stage in paras_stage:
                paras_array_name[f'{para_stage}_name'] = [f'{para_stage}{n + 1}' for n in range(self.stage)]

        self.status = self.insulators[paras_array_name['status_name']].values
        self.pattern = self.insulators[paras_array_name['pattern_name']].values
        self.Tc = self.insulators[paras_array_name['Tc_name']].values
        self.qmc = self.insulators[paras_array_name['qmc_name']].values
        self.Th = self.insulators[paras_array_name['Th_name']].values
        self.qmh = self.insulators[paras_array_name['qmh_name']].values
        self.Thick = self.insulators[paras_array_name['Thick_name']].values

        self.nit = self.insulators["nit"]  # tube number of the insulator
        self.location = self.insulators["io"]

        if self.location == 0:  # reaction occurs in the tube side
            self.Din = self.Dt.values  # self.insulator_para['Din']
            # self.heater = max(self.heater, (523 - 333) / self.insulator_para['Thick'] * 0.2 * np.pi * self.Dt / 3)
        else:
            self.Din = self.insulators[paras_array_name['Din_name']].values
        self.Do = self.Din + self.insulators[paras_array_name['Thick_name']].values * 2
        insulators = pd.DataFrame(index=np.arange(self.stage),
                                  columns=['status', 'pattern'] + paras_stage + ['nit', 'location'])

        for n in range(self.stage):
            insulators.loc[n, ['status', 'pattern'] + paras_stage] = [self.status[n], self.pattern[n], self.Din[n],
                                                                      self.Thick[n], self.Tc[n], self.qmc[n],
                                                                      self.Th[n], self.qmh[n]]
            insulators.iloc[n, -2:] = [self.nit, self.location]

        return insulators

    @staticmethod
    def mixer(F1, T1, F2, T2, P, species):
        """
        ideal mixer
        ref: Modelling, Estimation and Optimization of the Methanol Synthesis with Catalyst Deactivation
        :param F1: component of input gas 1, mol/s; ndarray
        :param T1: temperature of input gas 1, K
        :param F2: component of input gas 2, mol/s; ndarray
        :param T2: temperature of input gas 2, K
        :param P: pressure of input gas, bar
        :param species: component of input gas, list
        :return: molar flux of components, temperature
        """
        species = [i if i != 'carbon monoxide' else 'CO' for i in species]
        num = len(species)
        F_in = np.vstack((F1, F2))
        T_in = np.array([T1, T2])
        Pi_in = np.zeros((2, num))
        Pi_in[0] = P * F1 / np.sum(F1) * 1e5
        Pi_in[1] = P * F2 / np.sum(F2) * 1e5
        H_in = np.zeros((2, num))
        for i in range(2):
            for j in range(num):
                H_in[i, j] = PropsSI('HMOLAR', 'T', T_in[i], 'P', Pi_in[i, j], species[j]) if Pi_in[i, j] != 0 else 0
        H_t = np.sum(H_in * F_in)  # J/s

        F_out = F1 + F2
        Pi_out = P * F_out / np.sum(F_out) * 1e5
        H_o = np.zeros(num)
        H_diff = 100000
        if abs(T1 - T2) < 0.2:
            T_out = (T1 + T2) / 2
        else:
            for T in np.arange(min(T1, T2), max(T1, T2), 0.1):
                for i in range(num):
                    H_o[i] = PropsSI('HMOLAR', 'T', T, 'P', Pi_out[i], species[i]) if Pi_out[i] != 0 else 0
                H_o_t = np.sum(H_o * F_out)
                cal_diff = abs(H_o_t - H_t)
                if cal_diff < H_diff:
                    H_diff = cal_diff
                    T_out = T
                if cal_diff / H_t < 0.001:
                    T_out = T
                    break
        return F_out, T_out

    @staticmethod
    def mixer_real(F1, T1, F2, T2, P, species):
        """
        real mixer
        ref: Modelling, Estimation and Optimization of the Methanol Synthesis with Catalyst Deactivation
        :param F1: component of input gas 1, mol/s; ndarray
        :param T1: temperature of input gas 1, K
        :param F2: component of input gas 2, mol/s; ndarray
        :param T2: temperature of input gas 2, K
        :param P: pressure of input gas, bar
        :param species: component of input gas, list
        :return: molar flux of components, temperature
        """
        cal = VLEThermo(species)
        P = P * 1E5
        H_in = cal.cal_H(T1, P, F1) + cal.cal_H(T2, P, F2)
        F_out = F1 + F2
        H_diff = 100000
        if abs(T1 - T2) < 0.2:
            T_out = (T1 + T2) / 2
        else:
            for T in np.arange(min(T1 - 10, T2 - 10), max(T1 + 10, T2 + 10), 0.1):
                H_o = cal.cal_H(T, P, F_out)
                cal_diff = abs(H_o - H_in)
                if cal_diff < H_diff:
                    H_diff = cal_diff
                    T_out = T
                if cal_diff / H_in < 0.001 / 1e3:
                    T_out = T
                    break
        return F_out, T_out

    def reactor_metric(self, sim_res):
        """
        save the one-pass performance of reactor
        :param sim_res: molar flux of each component along the reactor
        """

        # reactor metric
        # y= z [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, F_Ar
        # react: F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, F_Ar
        # diff: F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, F_Ar
        # Tr, Tc, Th, P, q_react, q_diff, q_heater, h_diff]
        To_r = sim_res[self.comp_num * 3 + 1, -1]  # reactor output temperature
        r = (sim_res[1, 0] - sim_res[1, -1]) / sim_res[1, 0]  # CO2 conversion ratio
        r_H2 = (sim_res[2, 0] - sim_res[2, -1]) / sim_res[2, 0]  # H2 conversion ratio
        r_c = (sim_res[1, 0] - sim_res[1, -1] + sim_res[5, 0] - sim_res[5, -1]) / (
                sim_res[1, 0] + sim_res[5, 0])  # conversion ratio of Carbon
        dF_react_rwgs = sim_res[5][-1] - sim_res[5][0]  # amount of reaction CO2 to CO
        dF_react_ch3oh = (sim_res[1, 0] - sim_res[1][-1]) - dF_react_rwgs  # amount of reaction CO2 to CH3OH
        dF_react_h2o = dF_react_rwgs + dF_react_ch3oh  # amount of water produced
        s_react = dF_react_ch3oh / (sim_res[1, 0] - sim_res[1, -1])  # selectivity of reactions
        dH_react = sim_res[self.comp_num * 3 + 5, -1]
        dP = sim_res[self.comp_num * 3 + 4, -1] - sim_res[self.comp_num * 3 + 4, 0]

        # in-situ separation metric

        Tin_c = sim_res[self.comp_num * 3 + 2, -1]  # input water temperature in the cold side of insulator
        To_h = sim_res[self.comp_num * 3 + 3, -1]
        dF_diff_ch3oh = dF_react_ch3oh - (sim_res[3, -1] - sim_res[3, 0])  # amount of CH3OH condensed
        dF_diff_h2o = dF_react_h2o - (sim_res[4][-1] - sim_res[4][0])  # amount of H2O condensed
        sp_ch3oh = dF_diff_ch3oh / dF_react_ch3oh  # separation ratio of CH3OH
        sp_h2o = dF_diff_h2o / dF_react_h2o  # separation ratio of H2O
        N_CH3OH_H2O = dF_diff_ch3oh / dF_diff_h2o
        q_diff = sim_res[self.comp_num * 3 + 6, -1]
        q_heater = sim_res[self.comp_num * 3 + 7, -1]
        dH_diff = sim_res[self.comp_num * 3 + 8, -1]
        yield_CH3OH = dF_diff_ch3oh if 1 in self.status else dF_react_ch3oh  # mol/s
        eff = q_heater / 1000 / (yield_CH3OH * 32)  # kJ/g CH3OH
        res = pd.Series([r, r_c, r_H2, s_react, yield_CH3OH, dP, To_r,
                         Tin_c, To_h, dH_react, q_diff, q_heater, dH_diff, eff, sp_ch3oh, sp_h2o, N_CH3OH_H2O],
                        index=['conversion', "r_c", "r_H2", 'select', 'y_CH3OH', 'dP', 'To_r', "Tin_c", "To_h",
                               'q_react', 'q_diff', 'q_heater', "H_diff",
                               "eff", 'sp_CH3OH', 'sp_H2O', 'N_CH3OH_H2O'])
        return res

    def recycle_metric(self, sim_profile, F_recycle, T_feed):
        """
        calculate the metric for the whole plant with recycled stream
        :param sim_profile: sim profile for one pass
        :param F_recycle: recycled gas
        :param T_feed: feed temperature
        :return:
        """

        # [L1, F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, Tr, Tc, q_react, q_diff]
        # calculate metric for recycled reactor, recycled ratio and enthalpy
        ratio = np.sum(F_recycle) / self.Ft0
        Tr_out = sim_profile[19, -1]

        cal = VLEThermo(self.comp_list)
        H_out = cal.cal_H(Tr_out, self.P0, sim_profile[1:self.comp_num + 1, -1])  # kW
        H_feed = cal.cal_H(T_feed, self.P0, self.F0)  # kW
        H_in = cal.cal_H(self.T0, self.P0, self.F0)  # kW
        H_recycled = cal.cal_H(self.T0, self.P0, sim_profile[1:self.comp_num + 1, -1])  # kW

        heat_duty = H_in - H_feed
        heat_recycle = H_recycled - H_out
        delta_H = heat_recycle + heat_duty

        p_metric = pd.Series([ratio, heat_recycle, heat_duty, delta_H],
                             index=['ratio', 'heat_recycle', 'duty', 'delta_H'])
        return p_metric

    def one_pass(self, reactor, insulator, feed, diff_in=None):
        """
        simulation of one-pass reaction for CO2 to CH3OH
        :param diff_in: initial condition for diffusion para
        :param feed: feed para
        :param insulator: insulator para, pd.Seize
        :param reactor: reactor paras, pd.Seize
        :return: molar flux and temperature profile along the reactor
        """
        [T_in, P_in, F_in] = feed.loc[['T0', 'P0', 'F0']].values
        [L, Dt, Dc, nrt, phi, rhoc] = reactor.loc[['L', 'Dt', "Dc", 'nrt', 'phi', 'rhoc']].values
        [Din, thick, Tc_in, qmc, Th_in, qmh] = insulator.loc[['Din', 'Thick', 'Tc', 'qmc', 'Th', 'qmh']].values
        Dt = Din if self.location == 0 else Dt
        [status, pattern, nit, location] = insulator.loc[['status', 'pattern', 'nit', 'location']].values
        print(Din, thick, L)
        # if status == 1:
        #     q_h_guess = round((T_in - Tc_in) / thick * 0.36 * np.pi * Din * 0.52, 2)  # 1 for CO 0.2 for CO2; W/m
        # #     # L = round(min(1400 / q_h_guess, L), 2)
        # #     print(Din, thick, heater, L, q_h_guess)
        # #     # self.reactors_para['L2'] = L
        # else:
        #     q_h_guess = 0
        # print(q_h_guess * L)
        Do = Din + thick * 2
        if diff_in is None:
            q_react_in, q_diff_in, q_heater_in, h_diff_in = 0, 0, 0, 0
        else:
            [q_react_in, q_diff_in, q_heater_in, h_diff_in] = diff_in

        if len(F_in) == 5:
            comp = ["CO2", "H2", "Methanol", "H2O", "CO"]
        else:
            comp = ["CO2", "H2", "Methanol", "H2O", "CO", 'Ar']
        react_sim = Reaction(L, Dt, Dc, nrt, phi, rhoc, self.chem_para, T_in, P_in, F_in, comp, self.eos, qmh)
        insula_sim = Insulation(Do, Din, nit, location, comp)
        F_feed_pd = pd.Series(self.F0, index=self.comp_list)
        property_feed = mixture_property(self.T_feed, xi_gas=F_feed_pd / np.sum(self.F0), Pt=P_in)
        property_in_gas = mixture_property(T_in, xi_gas=pd.Series(F_in / np.sum(F_in), index=comp), Pt=P_in)

        # self.heater = max(q_h_guess, heater)
        Twcs = []
        Ft_mass = np.sum(self.F0) * property_feed['M']  # kg/s
        mass = np.array([0.0440098, 0.00201588, 0.03204216, 0.01801527, 0.0280101])  # kg/mol

        # def cal_cp(T):
        #     # cp_para = [[3.259, 1.356, 1.502, -2.374, 1.056],
        #     #            [2.883, 3.681, -0.772, 0.692, -0.213],
        #     #            [4.714, -6.986, 4.211, -4.443, 1.535],
        #     #            [4.395, -4.186, 1.405, -1.564, 0.632],
        #     #            [3.912, -3.913, 1.182, -1.302, 0.515]]
        #     cp_para = [[7.014903984,8.24973727,1428,6.305531671,588,-223.15,4726.85],
        #                [6.596207127,2.283366772,2466,0.8980605713,567.6,-23.15,1226.85],
        #                [9.375179134,20.99455431,1916.5,12.81503774,896.7,0,1226.85],
        #                [7.968615649,6.398681571,2610.5,2.124773096,1169,-173.15,2000],
        #                [6.952326359,2.095395051,3085.1,2.01951371,1538.2,-213.15,1226.85]]
        #     # cp_para = cp_para * np.array([1, 1e-3, 1e-5, 1e-8, 1e-11])
        #     T_array = np.array([1, T, T ** 2, T ** 3, T ** 4])
        #     cp = []
        #     for para in cp_para:
        #         cp.append(np.dot(para, T_array) * 8.314)
        #     return np.array(cp)

        def cal_cp(T):
            # cp_para = [[3.259, 1.356, 1.502, -2.374, 1.056],
            #            [2.883, 3.681, -0.772, 0.692, -0.213],
            #            [4.395, -4.186, 1.405, -1.564, 0.632],
            #            [4.714, -6.986, 4.211, -4.443, 1.535],
            #            [3.912, -3.913, 1.182, -1.302, 0.515]]
            cp_para = [[7.014903984, 8.24973727, 1428, 6.305531671, 588, -223.15, 4726.85],
                       [6.596207127, 2.283366772, 2466, 0.8980605713, 567.6, -23.15, 1226.85],
                       [9.375179134, 20.99455431, 1916.5, 12.81503774, 896.7, 0, 1226.85],
                       [7.968615649, 6.398681571, 2610.5, 2.124773096, 1169, -173.15, 2000],
                       [6.952326359, 2.095395051, 3085.1, 2.01951371, 1538.2, -213.15, 1226.85]]
            # cp_para = cp_para * np.array([1, 1e-3, 1e-5, 1e-8, 1e-11])
            cps = []
            for para in cp_para:
                cp = para[0] + para[1] * (para[2] / T / np.sinh(para[2] / T)) ** 2 + \
                     para[3] * (para[4] / T / np.cosh(para[4] / T)) ** 2
                cps.append(cp)
            return np.array(cps) * 4.184

        cp_m_in = property_in_gas['cp_m'] / property_in_gas["M"]
        cp_in = property_in_gas['cp_m']
        pro_cal = VLEThermo(self.comp_list)

        def model(z, y):
            # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, F_Ar
            # react: F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, F_Ar
            # diff: F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, F_Ar
            # Tr, Tc, Th, P, q_react, q_diff, q_heater, Twc]
            F = np.array(y[:self.comp_num])
            Ft = np.sum(F)
            xi = F / Ft
            yi = xi * mass / np.dot(xi, mass)

            Tr, Tc = y[self.comp_num * 3], y[self.comp_num * 3 + 1]
            Th = y[self.comp_num * 3 + 2]
            P = y[self.comp_num * 3 + 3]
            # print(z, Tr, xi)
            # simulation of reactor
            res_react = react_sim.balance(Tr, P, F)
            dP_dz = 0 if self.drop == 0 else react_sim.ergun(Tr, P, F) * 1e-5  # bar/m
            # Rr_in, Rr_o = react_sim.htr(Tr, P, F) # heat resistance of catalytic layer
            Rr = react_sim.htr(Tr, P, F)  # heat resistance of catalytic layer
            Rr_in, Rr_o = Rr / np.pi / Dc, Rr / np.pi / Dt  # 1/(W/m K)
            # convert reaction rate per length to per kg catalyst
            dl2dw = np.pi * ((Dt ** 2) / 4) * rhoc * phi
            if status == 1 and z > 0:
                # the module insulator is on
                # volume fraction of catalyst
                # r_v_ins_v_react = Do ** 2 * nit / Dt ** 2 / nrt if location == 1 else 0
                r_v_ins_v_react = Dc ** 2 / Dt ** 2 if qmh != 0 else 0  # r_v_ins_v_react = 0.08 for CO exp 1999

                coe_Tc = -1 if location == 1 else 1

                # HMT calculation
                # find the temperature of diffusion layer on the reaction side, Twc
                Rh_d0 = insula_sim.Rh_d_pre()
                Rh_c0 = Rh_d0 + Rr_o
                qh = (Th - Tr) / Rr_in if qmh != 0 else 0  # W/m
                qc0 = (Tr - Tc) / Rh_c0
                Twc0 = Tc + qc0 * Rh_d0
                Twc_cal = Twc0
                dev, n_iter = 100, 0
                dT_iter = 0.1
                # res_diff = insula_sim.flux(Twc_cal, P, F, Tc)
                while True:
                    res_diff_cal = insula_sim.flux(Twc_cal, P, F, Tc)  # simulation of insulator
                    qc_d = coe_Tc * res_diff_cal['hflux']
                    qc_r = (Tr - Twc_cal) / Rr_o
                    dev_cal = (abs(qc_d) - qc_r) / abs(qc_d)
                    if n_iter == 0:
                        dev_cal0 = dev_cal
                        res_diff = res_diff_cal
                        Twc = Twc_cal
                    if abs(dev_cal) < dev:
                        dev = abs(dev_cal)
                        Twc = Twc_cal
                        res_diff = res_diff_cal
                    if abs(dev_cal) < 0.01 or n_iter > 10:
                        if n_iter > 10:
                            print(n_iter, dev_cal, Twc, qc_d, qc_r, dT_iter)
                        break
                    if dev_cal0 < 0:
                        if dev_cal < 0:
                            Twc_cal += dT_iter
                        else:
                            Twc_cal -= dT_iter
                            dT_iter = max(0.05, dT_iter - 0.2)
                    else:
                        if dev_cal > 0:
                            Twc_cal -= dT_iter
                        else:
                            Twc_cal += dT_iter
                            dT_iter = max(0.05, dT_iter - 0.2)

                    n_iter += 1
                # print(n_iter, dev_cal, Twc, dT_iter)
                dTc_dz = -pattern * (coe_Tc * (res_diff['hflux'] + res_diff['hlg']) * nit) / qmc / 76
                dTh_dz = -qh / qmh / 86 if qmh != 0 else 0  # 513K 4MPa
            else:
                r_v_ins_v_react = 0
                res_diff = {'mflux': np.zeros(len(self.comp_list)), 'hflux': 0, "Tvar": 0, 'hlg': 0, 'ht': 0}
                dTc_dz = -pattern * self.Uc * (Tc - Tr) * np.pi * Dt / (property_feed["cp_m"] * np.sum(F_in))
                qh, dTh_dz, Twc, dev = 0, 0, 0, 0
            Twcs.append([z, Twc, dev, 1 / Rr_o, ks / thick * np.pi * Dt])
            cooler = self.Uc * (Tc - Tr) * np.pi * Dt  # res_react['tc']

            dF_react_dz = res_react['mflux'] * dl2dw * (1 - r_v_ins_v_react) * self.nrt
            dF_diff_dz = res_diff["mflux"] * nit
            dF_dz = dF_react_dz + dF_diff_dz
            dFt_dz = np.sum(dF_dz)
            dF_m_dz = dF_dz * mass
            dFt_m_dz = np.sum(dF_m_dz)
            # print(dFt_m_dz, dFt_dz)

            prop_m = mixture_property(Tr, pd.Series(F / Ft, index=self.comp_list), P)
            # cp = prop_m[5:].values
            # print(cp)
            cp = pro_cal.cal_cp_ig(Tr)#cal_cp(Tr)
            # print(cp)
            # cp_m = np.dot(cp, yi)
            # M_ave = np.array(xi, mass)
            # print(cp)
            # cp_m = np.dot(cp, xi)
            # # cp_m = prop_m['cp_m']
            # M_m = prop_m['M']
            # cps.append([z, cp_m, cp_m / M_m, M_m])
            cp_ave = prop_m['cp_m']
            cp_ave_mass = cp_ave / prop_m['M']  # J/kg K
            heat_cap = np.dot(cp, F)
            q_heater = qh  # q_h_guess  #
            # dTr_dz = (res_react["hflux"] * dl2dw * (1 - r_v_ins_v_react) * nrt +
            #           cooler + q_heater - res_diff['ht']) / heat_cap - np.dot(dF_dz, cp) * Tr / heat_cap
            # dTr_dz = (res_react["hflux"] * dl2dw * (1 - r_v_ins_v_react) * nrt +
            #           cooler + q_heater - res_diff['ht']) / (Ft*cp_ave) - dFt_dz * Tr / Ft
            # dTr_dz = 0
            dTr_dz = (res_react["hflux"] * dl2dw * (1 - r_v_ins_v_react) * nrt +
                      cooler + q_heater - res_diff['ht']) / heat_cap  # - np.dot(dF_dz, cp) * Tr / heat_cap  # - dFt_dz * Tr/Ft#(heat_cap) - dFt_dz * Tr / Ft  # (Ft_mass * cp_ave_mass)  # - dFt_dz * Tr / Ft
            # print(dFt_dz, dF_dz)
            # a = (res_react["hflux"] * dl2dw * (1 - r_v_ins_v_react) * nrt +
            #      cooler + q_heater - res_diff['ht']) / (Ft * cp_in)
            # b = dFt_dz * Tr / Ft  # (heat_cap)
            # print(a, b)
            dq_rea_dz = res_react["hflux"] * dl2dw * (1 - r_v_ins_v_react) * nrt  # W/m
            dq_dif_dz = res_diff['hflux'] * nit  # res_react['tc']  # W/m
            dh_dif_dz = res_diff['ht'] * nit

            res_dz = np.hstack((dF_dz, dF_react_dz, dF_diff_dz,
                                np.array([dTr_dz, dTc_dz, dTh_dz, dP_dz, dq_rea_dz, dq_dif_dz, q_heater, dh_dif_dz])))
            return res_dz

        z_span = [0, L]
        Tc_ini = Tc_in if status == 1 else self.T_feed
        Th_ini = Th_in

        ic = np.hstack((F_in, np.zeros(len(self.comp_list)), np.zeros(len(self.comp_list)),
                        np.array([T_in, Tc_ini, Th_ini, P_in, q_react_in, q_diff_in, q_heater_in, h_diff_in])))
        # LSODA BDF RK45 rtol=1e-9, max_step=0.0001
        res_sim = scipy.integrate.solve_ivp(model, z_span, ic, method='LSODA',
                                            t_eval=np.linspace(0, L, 1000), rtol=1e-6,max_step=0.01)
        if status == 1:
            Twcs_path = f"D:/document/04Code/PycharmProjects/MTC/result/Twc_{Dt}_{L}_{thick}_{Tc_ini}_{Th_ini}.txt"
            np.savetxt(Twcs_path, np.array(Twcs))
        # cp_path = f"D:/document/04Code/PycharmProjects/MTC/result/cp_{T_in}_{L}.txt"
        # np.savetxt(cp_path, np.array(cps))
        res = np.vstack((np.linspace(0, L, 1000), res_sim.y))
        return res

    def multi_reactor(self):
        res = {}
        feed_para, diff_para = self.feed0_para, None

        for n in range(self.stage):
            insulator_para = self.insulator.iloc[n]
            reactor_para = self.reactor_para.iloc[n]
            res[f'{n}'] = self.one_pass(reactor_para, insulator_para, feed_para, diff_para)
            F_out = res[f'{n}'][1:(self.comp_num + 1), -1].copy()
            r_temp = (feed_para['F0'][0] - F_out[0]) / feed_para['F0'][0]

            Tr_out = res[f'{n}'][self.comp_num * 3 + 1, -1].copy()
            Tc_out = res[f'{n}'][self.comp_num * 3 + 2, -1].copy()
            Th_out = res[f'{n}'][self.comp_num * 3 + 3, -1].copy()
            P_out = res[f'{n}'][self.comp_num * 3 + 4, -1].copy()
            diff_para = res[f'{n}'][self.comp_num * 3 + 5:self.comp_num * 3 + 9, -1].copy()
            feed_para = pd.Series([Tr_out, P_out, F_out], index=['T0', 'P0', 'F0'])

            print(Tr_out, r_temp)
        if self.stage >= 1:
            res_profile = res['0']
            custom_array = np.linspace(0, self.L[0], 1000).tolist()
            for i in np.arange(1, self.stage):
                res_profile = np.hstack((res_profile, res[f'{i}']))
                custom_array += list(np.linspace(0, self.L[i], 1000) + sum(self.L[:i]))
            res_profile[0] = custom_array

        return res_profile

    def recycler(self, ratio=0.99, loop='direct', rtol=0.05):
        """
        solve the recycle loop using Wegstein convergence method
        only support the one-stage reactor
        :param loop: direct loop means the recycled gas is mixed with fresh feed directly
        :param ratio: ratio of reacted gas used to recycle
        :param rtol: relative tolerance of recycler calculation
        :return:
        """
        if self.stage > 1:
            raise ValueError('recycler does not support stage larger than 1!')

        # read the fresh feed and guess a recycle feed
        status = self.status

        F_fresh, T_fresh = self.feed0_para['F0'], self.feed0_para['T0']
        P_in = self.P0
        F_re0, T_re0 = np.zeros_like(F_fresh), T_fresh
        r_guess = -4 / 40 * P_in + 13
        F_re0[:2] = F_fresh[:2] * 3 if status == 1 else F_fresh[:2] * r_guess
        F_re0[2] = 0.5 * F_fresh[0] * (1 - 0.6) if status == 1 else 0.005 * F_fresh[0] * (1 - 0.8)
        F_re0[3] = 0.5 * F_fresh[0] * (1 - 0.9) if status == 1 else 0.005 * F_fresh[0] * (1 - 0.8)  # 0
        F_re0[4] = F_fresh[0] * 2 * 0.15 if status == 1 else F_fresh[0] * 2 * 0.01
        F_re0[5] = F_fresh[5] * 1.2 if status == 1 else F_fresh[5] * 1.5
        # print(F_fresh, T_fresh)
        # update the recycle stream using Wegstein method
        # ref: Abrol, et. al, Computers & Chemical Engineering, 2012
        F_diff = 1e5
        while F_diff > rtol:
            # Tr, Tc, P = y[-6], y[-5], y[-4]
            # print(F_re0, T_re0)
            F_in0, T_in0 = self.mixer_real(F_fresh, T_fresh, F_re0, T_re0, self.P0, self.comp_list)
            # print(F_in0, T_in0)
            feed = pd.Series([T_in0, P_in, F_in0], index=['T0', 'P0', 'F0'])
            res0 = self.one_pass(self.reactor_para.iloc[0], self.insulator.iloc[0], feed)
            F_re_cal0 = res0[:, -1][1:7]  # * ratio
            if loop == 'direct':
                T_re_cal0 = res0[:, -1][17] if status == 1 else self.T0  # self.T0  #
            elif loop == 'indirect':
                T_re_cal0 = self.T0
            if status == 0:
                # separation through flash can
                f_cal = VLEThermo(self.comp_list)
                f_g, f_l, vf = f_cal.flash(T=40 + 273.15, P=P_in, x=F_re_cal0)
                F_re_cal0 = np.array(f_g) * np.sum(F_re_cal0) * vf * ratio
            else:
                F_re_cal0[2:4] = F_re_cal0[2:4]

            F_re1, T_re1 = F_re_cal0, T_re_cal0
            F_in1, T_in1 = self.mixer_real(F_fresh, T_fresh, F_re1, T_re1, self.P0, self.comp_list)
            feed = pd.Series([T_in1, P_in, F_in1], index=['T0', 'P0', 'F0'])
            res1 = self.one_pass(self.reactor_para.iloc[0], self.insulator.iloc[0], feed)
            F_re_cal1 = res1[:, -1][1:7]  # * ratio
            T_re_cal1 = res1[:, -1][17] if status == 1 else self.T0  # self.T0  #
            if status == 0:
                # separation through flash can
                f_cal = VLEThermo(self.comp_list)
                f_g, f_l, vf = f_cal.flash(T=40 + 273.15, P=P_in, x=F_re_cal1)
                F_re_cal1 = np.array(f_g) * np.sum(F_re_cal1) * vf * ratio
            else:
                F_re_cal1[2:4] = F_re_cal0[2:4]

            # convergence criteria
            diff = np.abs(F_re_cal1 - F_re1) / F_re1
            diff[np.isnan(diff)] = 0
            F_diff = np.max(diff)

            # update the stream
            w_F = (F_re_cal1 - F_re_cal0) / (F_re1 - F_re0)
            w_F[np.isnan(w_F)] = 0
            q_F = w_F / (w_F - 1)
            q_F[q_F > 0], q_F[q_F < -5] = 0, -5

            if loop == 'indirect':
                T_re0 = self.T0
            else:
                w_T = (T_re_cal1 - T_re_cal0) / (T_re1 - T_re0) if status == 1 else 0
                q_T = w_T / (w_T - 1)
                q_T = -0.5 if q_T > 0 else q_T
                q_T = -5 if q_T < -5 else q_T
                T_re0 = q_T * T_re1 + (1 - q_T) * T_re_cal1
            # print(q_F,F_re1, F_re_cal1)
            F_re0 = q_F * F_re1 + (1 - q_F) * F_re_cal1
            F_re0[F_re0 < 0] = 0

            # F_in_temp, T_in_temp = self.mixer(F_fresh, T_fresh, F_re0, T_re0, self.P0, self.comp_list)
            # r_CO_CO2 = F_re0[4] / (F_re0[0] + F_fresh[0])
            # F_re0[4] = (F_re0[0] + F_fresh[0]) * 0.1 if r_CO_CO2 < 0.1 else F_re0[4]
            print(F_diff)
        return res1, F_re_cal1

    def to_r(self, r_target):
        if self.stage > 2:
            raise ValueError('stage > 2 is not supported with given conversion')
        # guess a length
        Tc, Dd = self.insulator.iloc[1]['Tc'], self.insulator.iloc[1]['Thick']
        # Psat_CH3OH = PropsSI('P', 'T', Tc, 'Q', 1, 'Methanol')
        # Pr_CH3OH = 35E5
        # Dt = self.reactor_para.iloc[1]['Dt']
        # N0_CH3OH = (Pr_CH3OH/503-Psat_CH3OH/Tc)/8.314/Dd*7.4E-7*np.pi*Dt
        N0_CH3OH = 1.4E-3 * 8E-3 / Dd
        L0 = round(0.008154456 / N0_CH3OH * r_target, 2) * 1.5  # 0.7 if self.reactors_para['Dt2'] > 0.06 else 2
        r_sim = 0
        L0_coe_pre, L0_coe = 1, 1
        n_iter = 0
        while r_sim < r_target or r_sim > 1:
            while True:
                self.reactors_para['L2'] = self.L[1] = self.reactor_para.iloc[1]['L'] = round(L0, 2)
                try:
                    res_profile = self.multi_reactor()
                    # If the operation is successful, break out of the loop
                    break
                except (chemicals.exceptions.PhaseCountReducedError,
                        AttributeError, ZeroDivisionError, OscillationError, ValueError) as e:
                    # Print the caught exception
                    print(e)
                    # Adjust L0
                    dL = L0 * 0.05
                    L0 = L0 - max(dL, 0.01)
            r_metric = self.reactor_metric(res_profile)
            r_sim = r_metric['conversion']
            print(r_sim, n_iter)
            # detect the improper configuration
            if L0 > 15:
                # print(f"ValueError: 'too low r too long reactor!':{r_sim:.2f}_{L0:.2f}")
                return res_profile
                # raise ValueError(f"{RED}ValueError: 'too low r too long reactor!':{r_sim:.2f}_{L0:.2f}{ENDC}")
            if res_profile[17, -1] < 443:
                return res_profile
                # print(f"ValueError: 'too low heater!':{r_sim:.2f}_{L0:.2f}_{res_profile[-6, -1]:.2f}")
                # raise ValueError(f'{RED}too low heater!:{r_sim:.2f}_{L0:.2f}_{res_profile[-6, -1]:.2f}{ENDC}')
            if n_iter > 10:
                return res_profile
            try:
                To = res_profile[17, -1]
            except UnboundLocalError:
                To = 483

            if r_sim < 0.4:
                L0_coe = (r_target / r_sim * 2) if To <= 463 else (r_target / r_sim * 1.4)
            elif 0.4 <= r_sim < 0.6:
                L0_coe = (r_target / r_sim * 1.5) if To <= 463 else (r_target / r_sim * 1.3)
            elif 0.6 <= r_sim < 0.7:
                L0_coe = (r_target / r_sim * 1.3) if To <= 463 else (r_target / r_sim * 1.1)
            elif 0.7 <= r_sim < 0.8:
                L0_coe = (r_target / r_sim * 1.01)  # - n_iter * 0.1
            elif 0.8 <= r_sim < 0.9:
                dL = ((r_target / r_sim * 1.02) * L0 - L0) if To <= 463 else ((r_target / r_sim * 0.9) * L0 - L0)
                L0 = L0 + dL if dL > 0.02 else L0 + 0.02
            elif 0.9 <= r_sim < 0.94:
                if To <= 463:
                    L0 *= (r_target / r_sim * 1.02)
                else:
                    L0 += 0.04
                # L0 -= n_iter*0.04
            elif 0.94 <= r_sim < 1:
                L0 += max((r_target / r_sim - 1) * L0, 0.05) if To <= 463 else max((r_target / r_sim - 1) * L0, 0.01)
            elif r_sim > 1:
                L0 *= 0.9
                print('too long reactor too overlarge r!')
                # raise Warning('too long reactor too overlarge r!')
            else:
                pass
            if r_sim < 0.8:
                dL = (L0_coe - 1) * L0
                if n_iter % 2 == 0:
                    dL = max(dL * 0.6, 0.1)
                L0 = L0 + dL
            n_iter += 1
        return res_profile

    def sim(self, save_profile=0, loop='direct', rtol=0.05, r_target=None):
        kn_model = self.chem_para['kn_model']
        loop = "direct" if self.recycle == 0 else loop
        if self.recycle == 1:
            res_profile, F_recycle = self.recycler(loop=loop, rtol=rtol)
            r_metric = self.reactor_metric(res_profile)
            T_feed = res_profile[17, 1000 - 1] if self.stage == 2 else self.T_feed
            # calculate metric for recycled reactor
            p_conversion = pd.Series(r_metric["y_CH3OH"] / self.F0[0], index=["p_conversion"])
            p_metric = self.recycle_metric(res_profile, F_recycle, T_feed)
            p_metric = pd.concat([p_metric, p_conversion])

            metric = pd.concat([p_metric, r_metric])
            res_path = 'result/sim_recycle_%s_%s_%.4f_log.xlsx' % (ks, kn_model, rtol)
        else:
            res_profile = self.multi_reactor() if r_target is None else self.to_r(r_target)
            r_metric = self.reactor_metric(res_profile)
            metric = r_metric
            res_path = 'result/sim_one_pass_U_%s_%s_%s_%s_log.xlsx' % (
                ks, kn_model, self.stage, datetime.now().date())
        # calculate partial fugacity along the reactor

        print("*" * 10)
        print(r_metric)

        # concat the input para with the performance metrics
        feed_cond = self.feed0_para
        res = pd.concat([self.reactors_para, self.insulators, feed_cond, metric])
        res_save = pd.DataFrame(res.values.reshape(1, len(res.values)), columns=res.index)

        # save data to the Excel
        try:
            with pd.ExcelWriter(res_path, engine='openpyxl', mode='a', if_sheet_exists="overlay") as writer:
                try:
                    res_saved = pd.read_excel(res_path, sheet_name=loop)
                    res_save = pd.concat([res_saved, res_save], ignore_index=True)
                    # print(res_save)
                    # res_save = res_saved.append(res_save, ignore_index=True)
                    res_save.to_excel(writer, index=False, header=True, sheet_name=loop)
                except ValueError:
                    res_save.to_excel(writer, index=False, header=True, sheet_name=loop)
        except FileNotFoundError:
            res_save.to_excel(res_path, index=False, header=True, sheet_name=loop)
        if save_profile == 1:
            save_data = pd.DataFrame(res_profile.T, columns=['z'] + self.comp_list +
                                                            ['dF_re_' + i for i in self.comp_list] +
                                                            ['dF_diff_' + i for i in self.comp_list] +
                                                            ['Tr', 'Tc', 'Th', 'dP',
                                                             'q_react', 'q_diff', 'q_heater', 'h_diff'])
            # sim_path = 'result/sim_profile_U_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.xlsx' \
            #            % (self.stage, ks, kn_model, self.status, self.recycle, self.Dt.values, self.L.values,
            #               self.T0, self.P0, self.Tc, self.Th)
            sim_path = 'result/sim_validate.xlsx'
            # sim_path = 'result/sim_profile_U_%s_%s_%s_%s_%s_%s_%s_%s_%s.xlsx' \
            #            % (self.stage, ks, kn_model, self.recycle, self.L.values,
            #               self.T0, self.P0, self.Tc, self.Th)
            sheet_name = 'U_%s_eos_%s_drop_%s' % (self.Uc, self.eos, self.drop) if self.recycle == 0 else \
                f'U_{self.Uc}_drop_{self.drop}_rtol_{rtol}'
            try:
                with pd.ExcelWriter(sim_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                    save_data.to_excel(writer, index=False, sheet_name=sheet_name)
            except FileNotFoundError:
                save_data.to_excel(sim_path, index=False, sheet_name=sheet_name)
        return res_save
