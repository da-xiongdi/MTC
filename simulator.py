import os.path

import scipy

from insulator import Insulation, ks
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI


class Simulation(Insulation):
    def __init__(self, reactor_para, chem_para, feed_para, insulator_para):
        super(Simulation, self).__init__(reactor_para, chem_para, feed_para, insulator_para)
        self.status = self.insulator_para['status']
        self.Tc = self.insulator_para['Tc']
        self.qm_w = self.insulator_para['qm']
        self.q_v = self.insulator_para['q']

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
        # [L1, F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, Tr, Tc, q_react, q_diff]
        To_r = sim_res[6, -1]  # reactor output temperature
        r = (sim_res[1, 0] - sim_res[1, -1]) / sim_res[1, 0]  # conversion ratio
        dF_react_rwgs = sim_res[5][-1] - sim_res[5][0]  # amount of reaction CO2 to CO
        dF_react_ch3oh = (sim_res[1, 0] - sim_res[1][-1]) - dF_react_rwgs  # amount of reaction CO2 to CH3OH
        dF_react_h2o = dF_react_rwgs + dF_react_ch3oh  # amount of water produced
        s_react = dF_react_ch3oh / (sim_res[1, 0] - sim_res[1, -1])  # selectivity of reactions
        dH_react = sim_res[-2, -1]

        # in-situ separation metric
        v_react = (self.Dt ** 2 - self.Do ** 2) if self.location == 1 else self.Dt ** 2
        v_react = v_react / 4 * np.pi * self.L1  # catalyst volume
        area_cond = self.Do * np.pi * self.L1 if self.location == 1 else self.Dt * np.pi * self.L1  # condenser area
        ratio_v_area = v_react / area_cond
        Tin_c = sim_res[7, -1]  # input water temperature in the cold side of insulator
        dF_diff_ch3oh = dF_react_ch3oh - (sim_res[3, -1] - sim_res[3, 0])  # amount of CH3OH condensed
        dF_diff_h2o = dF_react_h2o - (sim_res[4][-1] - sim_res[4][0])  # amount of H2O condensed
        sp_ch3oh = dF_diff_ch3oh / dF_react_ch3oh  # separation ratio of CH3OH
        sp_h2o = dF_diff_h2o / dF_react_h2o  # separation ratio of H2O
        N_CH3OH_H2O = dF_diff_ch3oh / dF_diff_h2o
        dH_diff = sim_res[-1, -1]

        yield_CH3OH = dF_react_ch3oh if self.status == 0 else dF_diff_ch3oh
        res = pd.Series([ratio_v_area, r, s_react, yield_CH3OH, To_r,
                         Tin_c, dH_react, dH_diff, sp_ch3oh, sp_h2o, N_CH3OH_H2O],
                        index=["V/A", 'conversion', 'select', 'y_CH3OH', 'To_r', "Tin_c",
                               'q_react', 'q_diff', 'sp_CH3OH', 'sp_H2O', 'N_CH3OH_H2O'])
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

    def one_pass(self, status=None, L=None, F_in=None, T_in=None):
        """
        simulation of one-pass reaction for CO2 to CH3OH
        :param status: status of insulator, 1 or 0
        :param L: length of reactor, m
        :param F_in: reactor input gas, mol/s; ndarray
        :param T_in: reactor input temperature, K
        :return: molar flux and temperature profile along the reactor
        """
        if F_in is None:
            F_in = self.F0
            T_in = self.T0
        F_feed_pd = pd.Series(self.F0, index=self.comp_list)
        status = self.status if status is None else status
        L = self.L1 if L is None else L
        P = self.P0
        latent_heat = PropsSI('HMOLAR', 'P', 1e5, 'Q', self.q_v, "water") - PropsSI('HMOLAR', 'P', 1e5, 'Q', 0, "water")
        property_feed = self.mixture_property(self.T_feed, F_feed_pd / np.sum(self.F0) * self.P0)

        def model(z, y):
            # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, Tr, Tc, q_react, q_diff]
            F = np.array(y[:5])
            Tr, Tc = y[5], y[6]
            # simulation of reactor
            res_react = self.balance(Tr, P, F)
            # convert reaction rate per length to per kg catalyst
            dl2dw = np.pi * ((self.Dt ** 2) / 4) * self.rhoc * self.phi

            if status == 1:
                # the module insulator is on
                # volume fraction of catalyst
                r_v_ins_v_react = self.Do ** 2 * self.nit / self.Dt ** 2 / self.nrt if self.location == 1 else 0
                coe_Tc = -1 if self.location == 1 else 1
                res_diff = self.flux(Tr, P, F, Tc)  # simulation of insulator
                dTc_dz = (coe_Tc * res_diff['hflux'] * self.nit) / self.qm_w / 76 #\
                    # if abs(y[-1]) > (latent_heat * self.qm_w) else 0  # 0.8
            else:
                r_v_ins_v_react = 0
                res_diff = {'mflux': 0, 'hflux': 0, "Tvar": 0}
                dTc_dz = -self.Uc * (Tc - Tr) * np.pi * self.Dt / (property_feed["cp_m"] * np.sum(F_in))
            cooler = self.Uc * (Tc - Tr) * np.pi * self.Dt / res_react['tc']
            dF_dz = res_react['mflux'] * dl2dw * (1 - r_v_ins_v_react) * self.nrt + res_diff["mflux"] * self.nit
            dTr_dz = res_react["Tvar"] * dl2dw * (1 - r_v_ins_v_react) * self.nrt + res_diff["Tvar"] * self.nit + cooler

            dq_rea_dz = res_react["hflux"] * dl2dw * (1 - r_v_ins_v_react) * self.nrt
            dq_dif_dz = res_diff['hflux'] * self.nit + cooler * res_react['tc']
            return np.append(dF_dz, np.array([dTr_dz, dTc_dz, dq_rea_dz, dq_dif_dz]))

        z_span = [0, L]
        Tc_ini = self.Tc if status == 1 else self.T_feed
        ic = np.append(F_in, np.array([T_in, Tc_ini, 0, 0]))
        res_sim = scipy.integrate.solve_ivp(model, z_span, ic, method='LSODA', t_eval=np.linspace(0, L, 1000))
        res = np.vstack((np.linspace(0, L, 1000), res_sim.y))
        return res

    def dual_reactor(self, F, T):
        res_CR = self.one_pass(status=0, L=self.L1, F_in=F, T_in=T)
        F_out_CR, Tr_out_CR, Tc_out_CR = res_CR[1:6, -1], res_CR[6, -1], res_CR[7, -1]
        res_NR = self.one_pass(status=1, L=self.L2, F_in=F_out_CR, T_in=Tr_out_CR)
        res_profile = np.hstack((res_CR, res_NR))
        res_profile[0] = np.hstack((np.linspace(0, self.L1, 1000), np.linspace(0, self.L2, 1000) + self.L1))
        return res_profile

    def recycler(self, ratio=0.99, status=None, loop='direct', rtol=0.05):
        """
        solve the recycle loop using Wegstein convergence method
        :param loop: direct loop means the recycled gas is mixed with fresh feed directly
        :param ratio: ratio of reacted gas used to recycle
        :param status: status of insulator, 1 or 0
        :param rtol: relative tolerance of recycler calculation
        :return:
        """
        # read the fresh feed and guess a recycle feed
        status = self.status if status is None else status
        F_fresh, T_fresh = self.F0, self.T0
        F_re0, T_re0 = np.zeros_like(self.F0), self.T0
        F_re0[:2] = self.F0[:2] * 1.5
        F_re0[2] = 0.5 * self.F0[0] * (1 - 0.6) if status == 1 else 0
        F_re0[3] = 0.5 * self.F0[0] * (1 - 0.9) if status == 1 else 0  # 0
        F_re0[4] = self.F0[0] * 2 * 0.15 if status == 1 else self.F0[0] * 2 * 0.01
        print(F_fresh, T_fresh)
        # update the recycle stream using Wegstein method
        # ref: Abrol, et. al, Computers & Chemical Engineering, 2012
        F_diff = 1e5
        while F_diff > rtol:
            print(F_re0, T_re0)
            F_in0, T_in0 = self.mixer(F_fresh, T_fresh, F_re0, T_re0, self.P0, self.comp_list)
            print(F_in0, T_in0)
            res0 = self.one_pass(status=status, F_in=F_in0, T_in=T_in0) if self.stage == 0 else \
                self.dual_reactor(F_in0, T_in0)
            F_re_cal0 = res0[:, -1][1:6] * ratio
            if loop == 'direct':
                T_re_cal0 = res0[:, -1][6] if status == 1 else self.T0  # self.T0  #
            elif loop == 'indirect':
                T_re_cal0 = self.T0
            F_re_cal0[2:4] = 0 if status == 0 else F_re_cal0[2:4]

            F_re1, T_re1 = F_re_cal0, T_re_cal0
            F_in1, T_in1 = self.mixer(F_fresh, T_fresh, F_re1, T_re1, self.P0, self.comp_list)
            res1 = self.one_pass(status=status, F_in=F_in1, T_in=T_in1) if self.stage == 1 \
                else self.dual_reactor(F_in1, T_in1)
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

    def sim(self, save_profile=0, loop='direct', rtol=0.05):

        feed_cond = pd.Series(self.feed_para)
        reactor_cond = pd.Series(self.react_para)
        insulator_cond = pd.Series(self.insulator_para)
        feed_cond["H2"] = self.H2
        feed_cond["Sv"] = self.sv
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
            res_path = 'result/sim_recycle_%s_%s_%.2f_log.xlsx' % (ks, self.kn_model, rtol)
        else:
            res_profile = self.one_pass() if self.stage == 1 else self.dual_reactor(self.F0, self.T0)
            r_metric = self.reactor_metric(res_profile)
            metric = r_metric
            res_path = 'result/sim_one_pass_%s_%s_log.xlsx' % (ks, self.kn_model)

        print(r_metric)
        # concat the input para with the performance metrics
        res = pd.concat([reactor_cond, insulator_cond, feed_cond, metric])
        res_save = pd.DataFrame(res.values.reshape(1, len(res.values)), columns=res.index)

        # save data to the Excel
        try:
            with pd.ExcelWriter(res_path, engine='openpyxl', mode='a', if_sheet_exists="overlay") as writer:
                try:
                    res_saved = pd.read_excel(res_path, sheet_name=loop)
                    res_save = res_saved.append(res_save, ignore_index=True)
                    res_save.to_excel(writer, index=False, header=True, sheet_name=loop)
                except ValueError:
                    res_save.to_excel(writer, index=False, header=True, sheet_name=loop)
        except FileNotFoundError:
            res_save.to_excel(res_path, index=False, header=True, sheet_name=loop)
        if save_profile == 1 and r_metric["conversion"] > 0.15:
            save_data = pd.DataFrame(res_profile.T, columns=['z'] + self.comp_list + ['Tr', 'Tc', 'q_react', 'q_diff'])
            sim_path = 'result/sim_profile_%s_%s_%s_%s_%s_%s_%s_%s_%s_%.2f.xlsx' \
                       % (ks, self.kn_model, self.status, self.recycle, self.Dt, self.L1,
                          self.T0, self.P0, self.Tc, self.sv)
            sheet_name = 'U_%s_qm_%s_qv_%s_stage_%s' % (self.Uc, self.qm_w, self.q_v, self.stage)
            try:
                with pd.ExcelWriter(sim_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                    save_data.to_excel(writer, index=False, sheet_name=sheet_name)
            except FileNotFoundError:
                save_data.to_excel(sim_path, index=False, sheet_name=sheet_name)

# F_1 = np.array([0.331076816, 1.10260401, 0.043283834, 0.074548673, 0.062756616])
# F_2 = np.array([0.16302695,0.48908084,0,0,0])
# T_1 = 528
# T_2 = 503
# gas = ['CO2', 'H2', 'Methanol', 'H2O', 'CO']
# print(Simulation.mixer(F_1, T_1, F_2, T_2, 50, gas))

# a = np.array([1, 2, 3, 4])
# b = np.zeros_like(a)
# b[0:2] = a[2:]
# print(b)
# print(a)
