import os.path

import scipy

from insulator import Insulation
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI


class Simulation(Insulation):
    def __init__(self, reactor_para, chem_para, feed_para, insulator_para, r_CH3OH_H2O):
        super(Simulation, self).__init__(reactor_para, chem_para, feed_para, insulator_para, r_CH3OH_H2O)
        self.status = self.insulator_para['status']

    @staticmethod
    def new_path(path):
        try:
            f = open(path)
            f.close()
            path_split = os.path.split(path)
            new_file_name = os.path.splitext(path_split[1])[0] + '_1' + os.path.splitext(path_split[1])[-1]
            path = os.path.join(path_split[0], new_file_name)
        except FileNotFoundError:
            path = path
        return path

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
                # J/mol
                H_in[i, j] = PropsSI('HMOLAR', 'T', T_in[i], 'P', Pi_in[i, j], species[j]) if Pi_in[i, j] != 0 else 0
        H_t = np.sum(H_in * F_in)  # J

        F_out = F1 + F2
        Pi_out = P * F_out / np.sum(F_out) * 1e5
        H_o = np.zeros(num)
        H_diff = 100000
        for T in np.arange(min(T1, T2), max(T1, T2), 0.1):
            for i in range(num):
                H_o[i] = PropsSI('HMOLAR', 'T', T, 'P', Pi_out[i], species[i]) if Pi_out[i] != 0 else 0
            H_o_t = np.sum(H_o * F_out)
            cal_diff = abs(H_o_t - H_t)
            if cal_diff < H_diff:
                H_diff = cal_diff
                T_out = T
            if cal_diff / H_t < 0.01:
                T_out = T
                break
        return F_out, T_out

    def save_data(self, sim_res, diff_res):

        # reactor performance
        To = sim_res[-1, -1]
        r = (sim_res[0, 0] - sim_res[0, -1]) / sim_res[0, 0]
        dF_react_rwgs = sim_res[4][-1] - sim_res[4][0]
        dF_react_ch3oh = (sim_res[0, 0] - sim_res[0][-1]) - dF_react_rwgs
        dF_react_h2o = dF_react_rwgs + dF_react_ch3oh
        dF_diff_ch3oh = dF_react_ch3oh - sim_res[2][-1]
        dF_diff_h2o = dF_react_h2o - sim_res[3][-1]
        sp_ch3oh = dF_diff_ch3oh / dF_react_ch3oh
        sp_h2o = dF_diff_h2o / dF_react_h2o
        N_CH3OH_H2O = dF_diff_ch3oh / dF_diff_h2o
        s_react = dF_react_ch3oh / (sim_res[0, 0] - sim_res[0, -1])

        reactor_cond = pd.Series(self.react_para)
        insulator_cond = pd.Series(self.insulator_para)
        feed_cond = pd.Series(self.feed_para)
        res = pd.Series([r, s_react, To, sp_ch3oh, sp_h2o, N_CH3OH_H2O, self.r_CH3OH_H2O],
                        index=['conversion', 'select', 'To', 'sp_CH3OH', 'sp_H2O', 'N_CH3OH_H2O', 'g_N_CH3OH_H2O'])

        res_save = pd.concat([feed_cond, reactor_cond, insulator_cond, res])
        res_save = pd.DataFrame(res_save.values.reshape(1, len(res_save.values)), columns=res_save.index)
        res_path = 'result/sim_recycle_%s_log.csv' % self.kn_model
        print(res)

        try:
            with open(res_path) as f:
                res_save.to_csv(res_path, mode='a', index=False, header=False)
        except FileNotFoundError:
            res_save.to_csv(res_path, mode='a', index=False, header=True)

        if r > 0.35:
            save_data = pd.DataFrame(sim_res.T, columns=self.comp_list + ['T'])
            sim_path = 'result/result_sim_%s_%s_%s_%s_%s_%s_%s_%s_%s.xlsx' \
                       % (self.kn_model, self.status, self.feed_para['CO/CO2'], self.Dt, self.L,
                          self.T0, self.P0, self.Tc, self.sv)
            # sim_path = self.new_path(sim_path)
            try:
                with pd.ExcelWriter(sim_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                    save_data.to_excel(writer)
            except FileNotFoundError:
                save_data.to_excel(sim_path)

            if self.status == 0:
                n_col = int(len(diff_res) / 14)
                save_performance = pd.DataFrame(np.array(diff_res).reshape((n_col, 14)))
                per_path = 'result/result_diff_%s_%s_%s_%s_%s_%s_%s_%s.xlsx' % \
                           (self.kn_model, self.status, self.Dt, self.L, self.T0, self.P0, self.Tc, self.sv)
                # per_path = self.new_path(per_path)
                try:
                    with pd.ExcelWriter(per_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                        save_performance.to_excel(writer)
                except FileNotFoundError:
                    save_performance.to_excel(per_path)

    def sin_pass(self, F_in, T_in):
        """
        simulation for CO2 TO CH3OH
        :param F_in: feed gas, mol/s; ndarray
        :param T_in: input temperature, K
        :return: concentration and its slop
        """

        P = self.P0
        performance = []

        def model(z, y):
            # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, T]
            F = np.array(y[:-1])
            T = y[-1]
            delta_react = self.balance(T, P, F)  # simulation of reactor
            # convert reaction rate per length to per kg catalyst
            dl2dw = np.pi * ((self.Dt ** 2) / 4) * self.rhoc * self.phi

            if self.status == 0:
                # the module insulator is on
                # volume fraction of catalyst
                r_v_ins_v_react = self.Do ** 2 * self.nit / self.Dt ** 2 / self.nrt if self.location == 'out' else 0
                delta_diff = self.flux(T, P, F)  # simulation of insulator
                # performance of diffusional module
                performance.append(z)
                performance.append(T)
                for i in (delta_react[0] * dl2dw).tolist(): performance.append(i)
                for i in (delta_diff[0] * self.nit).tolist(): performance.append(i)
                performance.append(delta_react[1] * dl2dw)
                performance.append(delta_diff[1] * self.nit)
            else:
                r_v_ins_v_react, delta_diff = 0, [0, 0]

            dF_dz = delta_react[0] * dl2dw * (1 - r_v_ins_v_react) * self.nrt + delta_diff[0] * self.nit
            dT_dz = delta_react[1] * dl2dw * (1 - r_v_ins_v_react) * self.nrt + delta_diff[1] * self.nit

            return np.append(dF_dz, [dT_dz])

        z_span = [0, self.L]
        ic = np.append(F_in, [T_in])
        # ic = np.append(self.F0, [self.T0])
        res = scipy.integrate.solve_ivp(model, z_span, ic, method='BDF', t_eval=np.linspace(0, self.L, 1000))
        data = res.y
        self.save_data(data, performance)

        return data, performance

    def recycler(self, ratio):
        F_fresh = self.F0
        T_fresh = self.T0
        F_in, T_in = self.F0, self.T0
        # [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, T]
        res = np.zeros((100, len(self.comp_list)+1))
        for n in range(100):
            res[n] = self.sin_pass(F_in, T_in)[0][:,-1]
            F_sin_out, T_sin_out = res[n][:-1], res[n][-1]
            F_diff_cal = np.abs(res[n][:-1] - res[n-1][:-1])/res[n-1][:-1]
            if np.max(F_diff_cal) < 0.01:
                break
            F_re = F_sin_out*ratio
            F_re[2:4] = 0
            F_in, T_in = self.mixer(F_fresh, T_fresh, F_re, T_sin_out, self.P0, self.comp_list)

        res_pd = pd.DataFrame(res, columns=['F_CO2', 'F_H2', 'F_CH3OH', 'F_H2O', 'F_CO', 'T'])
        res_pd.to_csv('result/recycle.csv')


# F_1 = np.array([0.078202587,0.207183322,0.021855522,0.007696604,0.018958482])
# F_2 = np.array([0.130682804,0.392048413,0,0,0.032670701])
# T_1 = 506.1447305
# T_2 = 533
# gas = ['CO2','H2','Methanol','H2O','CO']
# print(Simulation.mixer(F_1, T_1, F_2, T_2, 50, gas))
