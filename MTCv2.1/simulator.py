import scipy

from insulator import Insulation
import numpy as np
import pandas as pd


class Simulation(Insulation):
    def __init__(self):
        super().__init__()
        self.status = self.insulator_para['status']

    def save_data(self, sim_res, diff_res):

        # reactor performance
        To = sim_res[-1, -1]
        r = (sim_res[0, -1] - sim_res[0, 0]) / sim_res[0, 0]
        dF_react_rwgs = sim_res[4][-1] - sim_res[4][0]
        dF_react_ch3oh = (sim_res[0, 0] - sim_res[0][-1]) - dF_react_rwgs
        dF_react_h2o = dF_react_rwgs + dF_react_ch3oh
        dF_diff_ch3oh = dF_react_ch3oh - sim_res[2][-1]
        dF_diff_h2o = dF_react_h2o - sim_res[3][-1]
        sp_ch3oh = dF_diff_ch3oh / dF_react_ch3oh
        sp_h2o = dF_diff_h2o / dF_react_h2o
        r_CH3OH_H2O = dF_diff_ch3oh / dF_diff_h2o
        s_react = dF_react_ch3oh / (sim_res[0, -1] - sim_res[0, 0])

        reactor_cond = pd.Series(self.react_para)
        insulator_cond = pd.Series(self.insulator_para)
        feed_cond = pd.Series(self.feed_para)
        res = pd.Series([r, s_react, To, sp_ch3oh, sp_h2o, r_CH3OH_H2O],
                        index=['conversion', 'select', 'To', 'sp_CH3OH', 'sp_H2O', 'N_CH3OH_H2O'])
        res_save = pd.concat([feed_cond, reactor_cond, insulator_cond, res])
        res_save = pd.DataFrame(res_save.values.reshape(1, len(res_save.values)), columns=res_save.index)
        res_path = 'result/sim_log.csv'
        try:
            with open(res_path) as f:
                res_save.to_csv('result/sim_log.csv', mode='a', index=False, header=False)
        except FileNotFoundError:
            res_save.to_csv('result/sim_log.csv', mode='a', index=False, header=True)

        save_data = pd.DataFrame(sim_res.T, columns=self.comp_list + ['T'])
        sim_path = 'result/result_sim_%s_%s_%s_%s_%s_%s_%s.xlsx' \
                   % (self.status, self.Dt, self.L, self.T0, self.P0, self.Tc, self.v0)
        try:
            with pd.ExcelWriter(sim_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                save_data.to_excel(writer)
        except FileNotFoundError:
            save_data.to_excel(sim_path)

        if self.status == 'on':
            n_col = int(len(diff_res) / 14)
            save_performance = pd.DataFrame(np.array(diff_res).reshape((n_col, 14)))
            per_path = 'result/result_diff_%s_%s_%s_%s_%s_%s_%s.xlsx' % \
                       (self.status, self.Dt, self.L, self.T0, self.P0, self.Tc, self.v0)
            try:
                with pd.ExcelWriter(per_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                    save_performance.to_excel(writer)
            except FileNotFoundError:
                save_performance.to_excel(per_path)

    def simulator(self):
        """
        simulation for CO2 TO CH3OH
        :return: concentration and its slop
        """

        P = self.P0
        performance = []

        def model(z, y):
            # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, F_N2, T]
            F_in = np.array(y[:-1])
            T = y[-1]
            delta_react = self.balance(T, P, F_in)  # simulation of reactor
            # convert reaction rate per length to per kg catalyst
            dl2dw = np.pi * ((self.Dt ** 2) / 4) * self.rhoc * self.phi

            if self.status == 'on':
                # the module insulator is on
                # volume fraction of catalyst
                r_v_ins_v_react = self.Do ** 2 * self.nit / self.Dt ** 2 / self.nrt if self.location == 'out' else 0
                delta_diff = self.flux(T, P, F_in)  # simulation of insulator

                # performance of diffusional module
                performance.append(z)
                performance.append(T)
                for i in (delta_react[0] * dl2dw).tolist(): performance.append(i)
                for i in (delta_diff[0] * self.nit).tolist(): performance.append(i)
                performance.append(delta_react[1] * dl2dw)
                performance.append(delta_diff[1] * self.nit)
            else:
                r_v_ins_v_react, delta_react = 0, 0

            dF_dz = delta_react[0] * dl2dw * (1 - r_v_ins_v_react) * self.nrt + delta_diff[0] * self.nit
            dT_dz = delta_react[1] * dl2dw * (1 - r_v_ins_v_react) * self.nrt + delta_diff[1] * self.nit

            return np.append(dF_dz, [dT_dz])

        z_span = [0, self.L]
        ic = np.append(self.F0, [self.T0])
        res = scipy.integrate.solve_ivp(model, z_span, ic, method='BDF', t_eval=np.linspace(0, self.L, 1000))
        data = res.y
        self.save_data(data, performance)

        return data, performance


reactor = Simulation()
reactor.simulator()
# a = pd.Series([0, 1, 2], index=['a', 'b', 'c'])
# # # print(a.index.tolist())
# # a = pd.DataFrame(a.values.reshape(1,3),  columns=['a', 'b', 'c'])
# # print(a)
# # a.to_csv('result/t1.csv')
# path = "D:/document/04Code/PycharmProjects/MTC/MTCv2.1/result/result_diff_on_0.03_10_503_45_343_0.00196.xlsx"
#
# # a.to_excel(path)
# with pd.ExcelWriter(path,engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
#     # 写入数据
#     a.to_excel(writer)
