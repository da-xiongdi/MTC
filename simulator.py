import os.path
import warnings

import chemicals
import scipy

from prop_calculator import mixture_property, VLE
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
        self.comp_list = ["CO2", "H2", "Methanol", "H2O", "CO"]
        self.R = 8.314
        self.eos = eos  # 0 for ideal 1 for SRK
        self.drop = drop  # 1 for ergun 0 for zero drop

        # reactor insulator feed para
        self.reactors_para = reactors_para  #

        self.reactor_para = self._reactor_para()  # rewrite the reactor para
        self.chem_para = chem_para
        self.feed_para = feed_para
        self.feed0_para = self._feed_para()
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
        self.L, self.Dt = self.reactors_para[L_name], self.reactors_para[Dt_name]
        self.nrt = self.reactors_para['nrt']
        reactors = pd.DataFrame(index=np.arange(self.stage),
                                columns=['L', 'Dt'] + self.reactors_para.index[self.stage * 2:].tolist())
        for n in range(self.stage):
            reactors.loc[n, ['L', 'Dt']] = [self.L[n], self.Dt[n]]
            reactors.iloc[n, 2:] = self.reactors_para[2 * self.stage:]
        return reactors

    def _feed_para(self):
        """
        generate feed para
        """
        self.P0, self.T0 = self.feed_para["P"], self.feed_para["T"]  # P0 bar, T0 K
        self.T_feed = self.feed_para["T_feed"]

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
                self.H2 = self.F0[1] * 8.314 * 273.15 / 1e5 * 3600  # Nm3/h
            else:
                self.H2 = self.feed_para["H2"]
                self.F0[1] = self.H2 / 3600 * 1e5 / self.R / 273.15  # mol/s
                self.F0[0] = self.F0[1] / self.feed_para["H2/CO2"]
                self.F0[4] = self.F0[0] * self.feed_para['CO/CO2']
                self.Ft0 = np.sum(self.F0)
                self.v0 = self.Ft0 * self.R * self.T0 / (self.P0 * 1e5)
                self.sv = self.v0 * self.nrt * 3600 * 4 / self.L[0] / np.pi / self.Dt[0] ** 2
            # print(self.sv, self.H2)
        else:  # recycled stream
            self.F0 = self.feed_para[self.comp_list].to_numpy()
            self.Ft0 = np.sum(self.F0)
            self.v0 = self.Ft0 * self.R * self.T0 / (self.P0 * 1e5)
            self.sv = self.v0 * self.nrt * 3600 * 4 / self.L[0] / np.pi / self.Dt[0] ** 2
            self.H2 = self.F0[1] * self.R * 273.15 / 1E5
        feed = pd.Series([self.T0, self.P0, self.F0], index=['T0', 'P0', 'F0'])
        return feed

    def _insulator_para(self):
        """
        reconstruct insulator para
        :return: [[stage1, paras...], [stage2, paras...]] pd.Dataframe
        """
        paras_stage = ['Din', 'Thick', 'Tc', 'qm', 'heater']
        paras_array_name = {'status_name': [f'status{n + 1}' for n in range(self.stage)],
                            'pattern_name': [f'pattern{n + 1}' for n in range(self.stage)]}
        for n in range(self.stage):
            for para_stage in paras_stage:
                paras_array_name[f'{para_stage}_name'] = [f'{para_stage}{n + 1}' for n in range(self.stage)]

        self.status = self.insulators[paras_array_name['status_name']].values
        self.pattern = self.insulators[paras_array_name['pattern_name']].values
        self.Tc = self.insulators[paras_array_name['Tc_name']].values
        self.qm = self.insulators[paras_array_name['qm_name']].values
        self.heater = self.insulators[paras_array_name['heater_name']].values
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
                                                                      self.Thick[n], self.Tc[n], self.qm[n],
                                                                      self.heater[n]]
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

    def reactor_metric(self, sim_res):
        """
        save the one-pass performance of reactor
        :param sim_res: molar flux of each component along the reactor
        """

        # reactor metric
        # y= [F_CO2, F_H2, F_CH3OH, F_H2O,
        # react: F_CO, F_CO2, F_H2, F_CH3OH, F_H2O, F_CO,
        # diff: F_CO, F_CO2, F_H2, F_CH3OH, F_H2O, F_CO,
        # Tr, Tc, P, q_react, q_diff, q_heater]
        To_r = sim_res[-6, -1]  # reactor output temperature
        r = (sim_res[1, 0] - sim_res[1, -1]) / sim_res[1, 0]  # conversion ratio
        dF_react_rwgs = sim_res[5][-1] - sim_res[5][0]  # amount of reaction CO2 to CO
        dF_react_ch3oh = (sim_res[1, 0] - sim_res[1][-1]) - dF_react_rwgs  # amount of reaction CO2 to CH3OH
        dF_react_h2o = dF_react_rwgs + dF_react_ch3oh  # amount of water produced
        s_react = dF_react_ch3oh / (sim_res[1, 0] - sim_res[1, -1])  # selectivity of reactions
        dH_react = sim_res[-3, -1]
        dP = sim_res[-4, -1] - sim_res[-4, 0]

        # in-situ separation metric

        Tin_c = sim_res[-5, -1]  # input water temperature in the cold side of insulator
        dF_diff_ch3oh = dF_react_ch3oh - (sim_res[3, -1] - sim_res[3, 0])  # amount of CH3OH condensed
        dF_diff_h2o = dF_react_h2o - (sim_res[4][-1] - sim_res[4][0])  # amount of H2O condensed
        sp_ch3oh = dF_diff_ch3oh / dF_react_ch3oh  # separation ratio of CH3OH
        sp_h2o = dF_diff_h2o / dF_react_h2o  # separation ratio of H2O
        N_CH3OH_H2O = dF_diff_ch3oh / dF_diff_h2o
        dH_diff = sim_res[-2, -1]
        dH_heater = sim_res[-1, -1]
        yield_CH3OH = dF_diff_ch3oh if 1 in self.status else dF_react_ch3oh  # mol/s
        eff = dH_heater / 1000 / (yield_CH3OH * 32)  # kJ/g CH3OH
        res = pd.Series([r, s_react, yield_CH3OH, dP, To_r,
                         Tin_c, dH_react, dH_diff, dH_heater, eff, sp_ch3oh, sp_h2o, N_CH3OH_H2O],
                        index=['conversion', 'select', 'y_CH3OH', 'dP', 'To_r', "Tin_c",
                               'q_react', 'q_diff', 'q_heater', "eff", 'sp_CH3OH', 'sp_H2O', 'N_CH3OH_H2O'])
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
        Pi_feed = pd.Series(self.F0 / self.Ft0 * self.P0, index=self.comp_list)
        Pi_gas_out = pd.Series(sim_profile[1:6, -1] / np.sum(sim_profile[1:6, -1]) * self.P0, index=self.comp_list)
        num_comp = len(self.comp_list)
        Tr_out = sim_profile[6, -1]
        H_out, H_recycled, H_in, H_feed = np.zeros((4, num_comp))
        for i in range(num_comp):
            H_out[i] = PropsSI('HMOLAR', 'T', Tr_out, 'P', Pi_gas_out[i], self.comp_list[i]) \
                if Pi_gas_out[i] != 0 else 0
            H_feed[i] = PropsSI('HMOLAR', 'T', T_feed, 'P', Pi_feed[i], self.comp_list[i]) if Pi_feed[i] != 0 else 0
            H_in[i] = PropsSI('HMOLAR', 'T', self.T0, 'P', Pi_feed[i], self.comp_list[i]) if Pi_feed[i] != 0 else 0
            H_recycled[i] = PropsSI('HMOLAR', 'T', self.T0, 'P', Pi_gas_out[i], self.comp_list[i]) \
                if Pi_gas_out[i] != 0 else 0
        H_out_t = np.sum(H_out * sim_profile[1:6, -1])  # W
        H_recycled_t = np.sum(H_recycled * sim_profile[1:6, -1])
        H_in_t = np.sum(H_in * self.F0)  # W
        H_feed_t = np.sum(H_feed * self.F0)
        heat_duty = H_in_t - H_feed_t
        heat_recycle = H_recycled_t - H_out_t
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
        [L, Dt, nrt, phi, rhoc] = reactor.loc[['L', 'Dt', 'nrt', 'phi', 'rhoc']].values
        [Din, thick, Tc_in, qm, heater] = insulator.loc[['Din', 'Thick', 'Tc', 'qm', 'heater']].values
        Dt = Din if self.location == 0 else Dt
        [status, pattern, nit, location] = insulator.loc[['status', 'pattern', 'nit', 'location']].values

        if status == 1:
            q_h_guess = round((T_in - Tc_in) / thick * 0.2 * np.pi * Din * heater, 2)
            # L = round(min(1400 / q_h_guess, L), 2)
            print(Din, thick, heater, L, q_h_guess)
            # self.reactors_para['L2'] = L
        else:
            q_h_guess = 0

        Do = Din + thick * 2
        if diff_in is None:
            q_react_in, q_diff_in, q_heater_in = 0, 0, 0
        else:
            [q_react_in, q_diff_in, q_heater_in] = diff_in

        react_sim = Reaction(L, Dt, nrt, phi, rhoc, self.chem_para, T_in, P_in, F_in, self.eos)
        insula_sim = Insulation(Do, Din, nit, location)
        F_feed_pd = pd.Series(self.F0, index=self.comp_list)
        property_feed = mixture_property(self.T_feed, xi_gas=F_feed_pd / np.sum(self.F0), Pt=P_in)

        # self.heater = max(q_h_guess, heater)

        def model(z, y):
            # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO
            # react: F_CO2, F_H2, F_CH3OH, F_H2O, F_CO,
            # diff: F_CO2, F_H2, F_CH3OH, F_H2O, F_CO,
            # Tr, Tc, P, q_react, q_diff, q_heater]
            F = np.array(y[:5])
            Tr, Tc, P = y[-6], y[-5], y[-4]
            # simulation of reactor
            res_react = react_sim.balance(Tr, P, F)
            dP_dz = 0 if self.drop == 0 else react_sim.ergun(Tr, P, F) * 1e-5  # bar/m
            # convert reaction rate per length to per kg catalyst
            dl2dw = np.pi * ((Dt ** 2) / 4) * rhoc * phi
            if status == 1 and z > 0:
                # the module insulator is on
                # volume fraction of catalyst
                r_v_ins_v_react = Do ** 2 * nit / Dt ** 2 / nrt if location == 1 else 0
                # r_v_ins_v_react = 0.08 for CO exp 1999
                coe_Tc = -1 if location == 1 else 1
                res_diff = insula_sim.flux(Tr, P, F, Tc)  # simulation of insulator
                dTc_dz = -pattern * (coe_Tc * (res_diff['hflux'] + res_diff['hlg']) * nit) / qm / 76
            else:
                r_v_ins_v_react = 0
                res_diff = {'mflux': np.zeros(len(self.comp_list)), 'hflux': 0, "Tvar": 0, 'hlg': 0}
                dTc_dz = -pattern * self.Uc * (Tc - Tr) * np.pi * Dt / (property_feed["cp_m"] * np.sum(F_in))

            heat_cap = mixture_property(Tr, pd.Series(F / np.sum(F), index=self.comp_list), P)['cp_m'] * np.sum(F)
            cooler = self.Uc * (Tc - Tr) * np.pi * Dt / heat_cap  # res_react['tc']

            dF_react_dz = res_react['mflux'] * dl2dw * (1 - r_v_ins_v_react) * self.nrt
            dF_diff_dz = res_diff["mflux"] * nit
            dF_dz = dF_react_dz + dF_diff_dz

            q_heater = q_h_guess  # heater#round(max(q_h_guess, heater)) #heater # 0 if z < 0.1*0.08 else for CO exp
            dTr_dz = res_react["Tvar"] * dl2dw * (1 - r_v_ins_v_react) * nrt + res_diff["Tvar"] * self.nit + \
                     cooler + q_heater / heat_cap  # res_react['tc']
            dq_rea_dz = res_react["hflux"] * dl2dw * (1 - r_v_ins_v_react) * self.nrt  # W/m
            dq_dif_dz = res_diff['hflux'] * self.nit + cooler * heat_cap  # res_react['tc']  # W/m

            # print(Tr, res_react["Tvar"] * dl2dw * (1 - r_v_ins_v_react) * nrt, res_diff["Tvar"] * self.nit)
            res_dz = np.hstack((dF_dz, dF_react_dz, dF_diff_dz,
                                np.array([dTr_dz, dTc_dz, dP_dz, dq_rea_dz, dq_dif_dz, q_heater])))
            return res_dz

        z_span = [0, L]
        Tc_ini = Tc_in if status == 1 else self.T_feed
        ic = np.hstack((F_in, np.zeros(len(self.comp_list)), np.zeros(len(self.comp_list)),
                        np.array([T_in, Tc_ini, P_in, q_react_in, q_diff_in, q_heater_in])))
        res_sim = scipy.integrate.solve_ivp(model, z_span, ic, method='BDF',
                                            t_eval=np.linspace(0, L, 1000))  # LSODA BDF
        res = np.vstack((np.linspace(0, L, 1000), res_sim.y))
        return res

    def multi_reactor(self):
        res = {}
        feed_para, diff_para = self.feed0_para, None

        for n in range(self.stage):
            insulator_para = self.insulator.iloc[n]
            reactor_para = self.reactor_para.iloc[n]
            res[f'{n}'] = self.one_pass(reactor_para, insulator_para, feed_para, diff_para)
            F_out, Tr_out, Tc_out = res[f'{n}'][1:6, -1].copy(), res[f'{n}'][-6, -1].copy(), res[f'{n}'][-5, -1].copy()
            P_out = res[f'{n}'][-4, -1].copy()
            diff_para = res[f'{n}'][-3:, -1].copy()
            feed_para = pd.Series([Tr_out, P_out, F_out], index=['T0', 'P0', 'F0'])
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
        # read the fresh feed and guess a recycle feed
        status = self.status

        F_fresh, T_fresh = self.feed0_para['F0'], self.feed0_para['P0']
        P_in = self.P0
        F_re0, T_re0 = np.zeros_like(F_fresh), T_fresh
        F_re0[:2] = F_fresh[:2] * 1.5
        F_re0[2] = 0.5 * F_fresh[0] * (1 - 0.6) if status == 1 else 0
        F_re0[3] = 0.5 * F_fresh[0] * (1 - 0.9) if status == 1 else 0  # 0
        F_re0[4] = F_fresh[0] * 2 * 0.15 if status == 1 else F_fresh[0] * 2 * 0.01
        print(F_fresh, T_fresh)
        # update the recycle stream using Wegstein method
        # ref: Abrol, et. al, Computers & Chemical Engineering, 2012
        F_diff = 1e5
        while F_diff > rtol:
            print(F_re0, T_re0)
            F_in0, T_in0 = self.mixer(F_fresh, T_fresh, F_re0, T_re0, self.P0, self.comp_list)
            print(F_in0, T_in0)
            feed = pd.Series(np.array([T_in0, P_in, F_in0]), index=['T0', 'P0', 'F0'])
            res0 = self.one_pass(self.reactor_para.iloc[0], self.insulator.iloc[0], feed)
            F_re_cal0 = res0[:, -1][1:6] * ratio
            if loop == 'direct':
                T_re_cal0 = res0[:, -1][6] if status == 1 else self.T0  # self.T0  #
            elif loop == 'indirect':
                T_re_cal0 = self.T0
            F_re_cal0[2:4] = 0 if status == 0 else F_re_cal0[2:4]

            F_re1, T_re1 = F_re_cal0, T_re_cal0
            F_in1, T_in1 = self.mixer(F_fresh, T_fresh, F_re1, T_re1, self.P0, self.comp_list)
            feed = pd.Series(np.array([T_in1, P_in, F_in1]), index=['T0', 'P0', 'F0'])
            res1 = self.one_pass(self.reactor_para.iloc[0], self.insulator.iloc[0], feed)
            F_re_cal1 = res1[:, -1][1:6] * ratio
            T_re_cal1 = res1[:, -1][6] if status == 1 else self.T0  # self.T0  #
            F_re_cal1[2:4] = 0 if status == 0 else F_re_cal1[2:4]

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

        return res1, F_re_cal1

    def to_r(self, r_target):
        if self.stage > 2:
            raise ValueError('stage > 2 is not supported with given conversion')
        # guess a length
        L0, r_sim = 1.2, 0
        Din, Dd, Tc_in, heater = self.Din[1], self.insulators['Thick2'], self.Tc[1], self.heater[1]
        q_h = round((500 - Tc_in) / Dd * 0.2 * np.pi * Din * heater, 2)

        while r_sim < r_target or r_sim > 1:
            self.reactors_para['L2'] = self.L[1] = self.reactor_para.iloc[1]['L'] = round(L0, 2)
            try:
                res_profile = self.multi_reactor()
            except (chemicals.exceptions.PhaseCountReducedError, AttributeError):
                L0 *= 0.95
            r_metric = self.reactor_metric(res_profile)
            r_sim = r_metric['conversion']
            print(r_sim)
            # detect the improper configuration
            if L0 > 15:
                # print(f"ValueError: 'too low r too long reactor!':{r_sim:.2f}_{L0:.2f}")
                return res_profile
                # raise ValueError(f"{RED}ValueError: 'too low r too long reactor!':{r_sim:.2f}_{L0:.2f}{ENDC}")
            if res_profile[-6, -1] < 460:
                return res_profile
                # print(f"ValueError: 'too low heater!':{r_sim:.2f}_{L0:.2f}_{res_profile[-6, -1]:.2f}")
                # raise ValueError(f'{RED}too low heater!:{r_sim:.2f}_{L0:.2f}_{res_profile[-6, -1]:.2f}{ENDC}')

            if r_sim < 0.4:
                L0 *= (r_target / r_sim * 1.3) if q_h <= 500 else (r_target / r_sim * 1.2)
            elif 0.4 <= r_sim < 0.6:
                L0 *= (r_target / r_sim * 1.2) if q_h <= 500 else (r_target / r_sim * 1.1)
            elif 0.6 <= r_sim < 0.7:
                L0 *= (r_target / r_sim * 1.1) if q_h <= 500 else (r_target / r_sim * 1.05)
            elif 0.7 <= r_sim < 0.8:
                L0 *= (r_target / r_sim * 1.04)
            elif 0.8 <= r_sim < 0.9:
                L0 *= (r_target / r_sim * 1.02)
            elif 0.9 <= r_sim < 1:
                L0 += max((r_target / r_sim - 1) * L0, 0.01) if q_h >= 800 else max((r_target / r_sim - 1) * L0, 0.05)
            elif r_sim > 1:
                L0 -= 0.4
                print('too long reactor too overlarge r!')
                # raise Warning('too long reactor too overlarge r!')
            else:
                pass

        return res_profile

    def sim(self, save_profile=0, loop='direct', rtol=0.05, r_target=None):
        kn_model = self.chem_para['kn_model']
        loop = "direct" if self.recycle == 0 else loop
        if self.recycle == 1:
            res_profile, F_recycle = self.recycler(loop=loop, rtol=rtol)
            r_metric = self.reactor_metric(res_profile)
            T_feed = res_profile[7, 1000 - 1] if self.stage == 2 else self.T_feed
            # calculate metric for recycled reactor
            p_conversion = pd.Series(r_metric["y_CH3OH"] / self.F0[0], index=["p_conversion"])
            p_metric = self.recycle_metric(res_profile, F_recycle, T_feed)
            p_metric = p_metric.append(p_conversion)

            metric = pd.concat([p_metric, r_metric])
            res_path = 'result/sim_recycle_%s_%s_%.2f_log.xlsx' % (ks, kn_model, rtol)
        else:
            res_profile = self.multi_reactor() if r_target is None else self.to_r(r_target)
            r_metric = self.reactor_metric(res_profile)
            metric = r_metric
            res_path = 'result/sim_one_pass_%s_%s_%s_%s_log.xlsx' % (ks, kn_model, self.stage, datetime.now().date())
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
                                                            ['Tr', 'Tc', 'dP', 'q_react', 'q_diff', 'q_heater'])
            sim_path = 'result/sim_profile_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.xlsx' \
                       % (self.stage, ks, kn_model, self.status, self.recycle, self.Dt.values, self.L.values,
                          self.heater, self.T0, self.P0, self.Tc)
            sheet_name = 'U_%s_eos_%s_drop_%s' % (self.Uc, self.eos, self.drop)
            try:
                with pd.ExcelWriter(sim_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                    save_data.to_excel(writer, index=False, sheet_name=sheet_name)
            except FileNotFoundError:
                save_data.to_excel(sim_path, index=False, sheet_name=sheet_name)
        return res_save
