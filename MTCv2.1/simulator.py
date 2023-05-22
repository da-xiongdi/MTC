import os.path

import scipy

from insulator import Insulation
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI


class Simulation(Insulation):
    def __init__(self, reactor_para, chem_para, feed_para, insulator_para):
        super(Simulation, self).__init__(reactor_para, chem_para, feed_para, insulator_para)
        self.status = self.insulator_para['status']

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
        H_t = np.sum(H_in * F_in)  # J

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
    def reactor_metric(sim_res):
        """
        save the one-pass performance of reactor
        :param sim_res: molar flux of each component along the reactor
        """

        # reactor metric
        # [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, T, q_react, q_diff]
        To = sim_res[-3, -1]  # reactor output temperature
        r = (sim_res[0, 0] - sim_res[0, -1]) / sim_res[0, 0]  # conversion ratio
        dF_react_rwgs = sim_res[4][-1] - sim_res[4][0]  # amount of reaction CO2 to CO
        dF_react_ch3oh = (sim_res[0, 0] - sim_res[0][-1]) - dF_react_rwgs  # amount of reaction CO2 to CH3OH
        dF_react_h2o = dF_react_rwgs + dF_react_ch3oh  # amount of water produced
        s_react = dF_react_ch3oh / (sim_res[0, 0] - sim_res[0, -1])  # selectivity of reactions
        dH_react = sim_res[-2, -1]

        # in-situ separation metric
        dF_diff_ch3oh = dF_react_ch3oh - sim_res[2][-1]  # amount of CH3OH condensed
        dF_diff_h2o = dF_react_h2o - sim_res[3][-1]  # amount of H2O condensed
        sp_ch3oh = dF_diff_ch3oh / dF_react_ch3oh  # separation ratio of CH3OH
        sp_h2o = dF_diff_h2o / dF_react_h2o  # separation ratio of H2O
        N_CH3OH_H2O = dF_diff_ch3oh / dF_diff_h2o
        dH_diff = sim_res[-1, -1]
        res = pd.Series([r, s_react, To, dH_react,dH_diff, sp_ch3oh, sp_h2o, N_CH3OH_H2O],
                        index=['conversion', 'select', 'To', 'q_react', 'q_diff', 'sp_CH3OH', 'sp_H2O', 'N_CH3OH_H2O'])
        return res

    def plant_metric(self, F_recycle, F_react_out):
        """
        save performance of the plant through the whole process
        :param F_recycle: recycled gas
        :param F_react_out: gas in the output of reactor
        """

        # calculate the performance
        ratio = np.sum(F_recycle) / self.Ft0  # recycle ratio
        F_react_in = F_recycle + self.F0  # gas in the input of reactor
        F_out = F_react_in - F_react_out  # products of the reactor
        res = pd.Series(np.append(F_out, ratio), index=self.comp_list + ['ratio'])

        return res

    def one_pass(self, F_in=None, T_in=None):
        """
        simulation of one-pass reaction for CO2 to CH3OH
        :param F_in: reactor input gas, mol/s; ndarray
        :param T_in: reactor input temperature, K
        :return: molar flux and temperature profile along the reactor
        """
        if F_in is None:
            F_in = self.F0
            T_in = self.T0
        P = self.P0

        def model(z, y):
            # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, T, q_react, q_diff]
            F = np.array(y[:5])
            T = y[5]
            # q = y[-1]
            # simulation of reactor
            res_react = self.balance(T, P, F)
            # convert reaction rate per length to per kg catalyst
            dl2dw = np.pi * ((self.Dt ** 2) / 4) * self.rhoc * self.phi

            if self.status == 1:
                # the module insulator is on
                # volume fraction of catalyst
                r_v_ins_v_react = self.Do ** 2 * self.nit / self.Dt ** 2 / self.nrt if self.location == 'out' else 0
                res_diff = self.flux(T, P, F)  # simulation of insulator
            else:
                r_v_ins_v_react = 0
                res_diff = {'mflux': 0, 'hflux': 0, "Tvar": 0}

            dF_dz = res_react['mflux'] * dl2dw * (1 - r_v_ins_v_react) * self.nrt + res_diff["mflux"] * self.nit
            dT_dz = res_react["Tvar"] * dl2dw * (1 - r_v_ins_v_react) * self.nrt + res_diff["Tvar"] * self.nit
            dq_rea_dz = res_react["hflux"] * dl2dw * (1 - r_v_ins_v_react) * self.nrt
            dq_dif_dz = res_diff['hflux'] * self.nit
            return np.append(dF_dz, np.array([dT_dz, dq_rea_dz, dq_dif_dz]))

        z_span = [0, self.L]
        ic = np.append(F_in, np.array([T_in, 0, 0]))
        res = scipy.integrate.solve_ivp(model, z_span, ic, method='BDF', t_eval=np.linspace(0, self.L, 1000))

        return res.y

    def recycler(self, ratio=0.99):
        """
        solve the recycle loop using Wegstein convergence method
        :param ratio: ratio of reacted gas used to recycle
        :return:
        """
        # read the fresh feed and guess a recycle feed
        F_fresh, T_fresh = self.F0, self.T0
        F_re0, T_re0 = np.zeros_like(self.F0), self.T0
        F_re0[:2] = self.F0[:2] * 2
        F_re0[2:4] = 0.4 * self.F0[0] * (1 - 0.95) if self.status == 1 else 0  # 0
        F_re0[4] = self.F0[0] * 2 * 0.1 if self.status == 1 else self.F0[0] * 2 * 0.01

        # update the recycle stream using Wegstein method
        # ref: Abrol, et. al, Computers & Chemical Engineering, 2012
        F_diff = 1e5
        while F_diff > 0.1:
            F_in0, T_in0 = self.mixer(F_fresh, T_fresh, F_re0, T_re0, self.P0, self.comp_list)
            res0 = self.one_pass(F_in0, T_in0)
            F_re_cal0 = res0[:, -1][:5] * ratio
            T_re_cal0 = res0[:, -1][5] if self.status == 1 else self.T0
            F_re_cal0[2:4] = 0 if self.status == 0 else F_re_cal0[2:4]

            F_re1, T_re1 = F_re_cal0, T_re_cal0
            F_in1, T_in1 = self.mixer(F_fresh, T_fresh, F_re1, T_re1, self.P0, self.comp_list)
            res1 = self.one_pass(F_in1, T_in1)
            F_re_cal1 = res1[:, -1][:5] * ratio
            T_re_cal1 = res1[:, -1][5] if self.status == 1 else self.T0
            F_re_cal1[2:4] = 0 if self.status == 0 else F_re_cal1[2:4]

            # convergence criteria
            diff = np.abs(F_re_cal1 - F_re1) / F_re1
            diff[np.isnan(diff)] = 0
            F_diff = np.max(diff)

            # update the stream
            w_F = (F_re_cal1 - F_re_cal0) / (F_re1 - F_re0)
            w_F[np.isnan(w_F)] = 0
            q_F = w_F / (w_F - 1)
            q_F[q_F > 0], q_F[q_F < -5] = -0.5, -5

            w_T = (T_re_cal1 - T_re_cal0) / (T_re1 - T_re0) if self.status == 1 else 0
            q_T = w_T / (w_T - 1)
            q_T = -0.5 if q_T > 0 else q_T
            q_T = -5 if q_T < -5 else q_T
            F_re0 = q_F * F_re1 + (1 - q_F) * F_re_cal1
            T_re0 = q_T * T_re1 + (1 - q_T) * T_re_cal1

        return res1, F_re_cal1

    def sim(self, save_profile=0):

        feed_cond = pd.Series(self.feed_para)
        reactor_cond = pd.Series(self.react_para)
        insulator_cond = pd.Series(self.insulator_para)

        if self.recycle == 1:
            res_profile, F_recycle = self.recycler()
            p_metric = self.plant_metric(F_recycle, res_profile[:-1, -1])
            r_metric = self.reactor_metric(res_profile)
            metric = pd.concat([p_metric, r_metric])
            res_path = 'result/sim_recycle_%s_log.csv' % self.kn_model
        else:
            res_profile = self.one_pass()
            r_metric = self.reactor_metric(res_profile)
            metric = r_metric
            res_path = 'result/sim_one_pass_%s_log.csv' % self.kn_model

        print(r_metric)
        # concat the input para with the performance metrics
        res = pd.concat([reactor_cond, insulator_cond, feed_cond, metric])
        res_save = pd.DataFrame(res.values.reshape(1, len(res.values)), columns=res.index)

        # save data to the Excel
        try:
            with open(res_path) as f:
                res_save.to_csv(res_path, mode='a', index=False, header=False)
        except FileNotFoundError:
            res_save.to_csv(res_path, mode='a', index=False, header=True)
        if save_profile == 1:
            save_data = pd.DataFrame(res_profile.T, columns=self.comp_list + ['T','q_react','q_diff'])
            sim_path = 'result/sim_profile_%s_%s_%s_%s_%s_%s_%s_%s_%s.xlsx' \
                       % (self.kn_model, self.status, self.recycle, self.Dt, self.L,
                          self.T0, self.P0, self.Tc, self.sv)
            try:
                with pd.ExcelWriter(sim_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                    save_data.to_excel(writer)
            except FileNotFoundError:
                save_data.to_excel(sim_path)

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
