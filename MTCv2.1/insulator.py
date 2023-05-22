import matplotlib.pyplot as plt

from reactor import Reaction, vof, ks
import numpy as np
import pandas as pd
import scipy
from CoolProp.CoolProp import PropsSI
from vle import VLE

R = 8.314


class Insulation(Reaction):
    def __init__(self, reactor_para, chem_para, feed_para, insulator_para):
        super(Insulation, self).__init__(reactor_para, chem_para, feed_para)

        # insulator parameters
        self.insulator_para = insulator_para
        self.nit = self.insulator_para["nit"]  # tube number of the insulator
        self.location = self.insulator_para["io"]
        self.Din = self.Dt  # self.insulator_para['Din']
        # print(self.Din)
        self.Do = self.Din + self.insulator_para['Thick'] * 2
        self.Tc = self.insulator_para['Tc']

    @staticmethod
    def cold_comp(Pi_in, P_sat_out):
        """
        calculate the partial pressure of each gas in the cold side
        :param Pi_in: partial pressure of each gas in the hot side, bar;ndarray
        :param P_sat_out: saturated pressure of condensates in the cold side, bar;ndarray
        :return: partial pressure of each gas at Tc, bar; ndarray
        """
        P_t = np.sum(Pi_in)
        xi_in = Pi_in / P_t
        xi_ncon = np.delete(xi_in, [2, 3])
        xi_ncon = xi_ncon / np.sum(xi_ncon)
        Pi_ncon_out = (P_t - np.sum(P_sat_out)) * xi_ncon
        Pi_out = np.insert(Pi_ncon_out, 2, P_sat_out)
        return Pi_out

    @staticmethod
    def p_sat(T, xi, dx=0.01):
        """
        calculate the saturated pressure of condensates for liquid with specific ratio
        :param T: temperature, K
        :param xi: molar ratio of CH3OH in liquid
        :param dx: step size of xi in the calculation
        :return: saturated pressure of condensate, bar; ndarray
        """
        mix_liquid = 'HEOS::Methanol[%s]&H2O[%s]' % (xi[0], xi[1])
        Pl_sat = PropsSI('P', 'T', T, 'Q', 0, mix_liquid)
        diff_x = 1e5
        for xg_H2O in np.arange(0, 1, dx):
            mix_gas = 'HEOS::Methanol[%s]&H2O[%s]' % (1 - xg_H2O, xg_H2O)
            Pv_sat = PropsSI('P', 'T', T, 'Q', 1, mix_gas)
            temp = abs(Pl_sat - Pv_sat)
            if temp < diff_x:
                xg_H2O_sel = xg_H2O
                diff_x = temp
                if diff_x / Pl_sat < 0.05: break
        pi_v = Pv_sat * np.array([1 - xg_H2O_sel, xg_H2O_sel]) * 1e-5
        return pi_v

    @staticmethod
    def ode_single(inner_cond, outer_cond, P, properties):
        """
        ode for the concentration distribution along the channel, only one condensate
        :param inner_cond: temperature, molar fraction, and radius at inside;list
        :param outer_cond: temperature, molar fraction, and radius at outside; list
        :param P: pressure of mixture, bar
        :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
        :return: concentration and its slop
        """

        P = P * 1e5
        D_ca, D_cb = 7.1e-6, 1.1e-5
        D_dm = 9.9e-5  # 1 / (0.25 / D_da + 0.75 / D_db)
        [cp, D, k] = properties
        [T1, c1, r1] = inner_cond
        [T2, c2, r2] = outer_cond

        def model(z, y):
            [x, N, T, dTdz] = y  # [-,mol/s/m2, K, K/m]
            dx_dz = -N * (1 - x) / D_dm / (P / R / T)
            dN_dz = -N / z
            d2T_dz2 = -dTdz / z + N * cp * dTdz / k
            return np.vstack((dx_dz, dN_dz, dTdz, d2T_dz2))

        def bound(ya, yb):
            return np.array([ya[0] - c1, ya[2] - T1, yb[0] - c2, yb[2] - T2])

        xa, xb = r1, r2
        xini = np.linspace(xa, xb, 11)
        yini = np.zeros((4, xini.size))
        yini[0] = np.linspace(c1, c2, xini.size)
        yini[1] = -D_dm * (2 * P / R / (T1 + T2)) * (c1 - c2) / (r1 - r2)
        yini[2] = np.linspace(T1, T2, xini.size)
        yini[3] = (T1 - T2) / (r1 - r2)
        res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-10, max_nodes=1000)
        xsol = np.linspace(xa, xb, 200)
        ysol = res.sol(xsol)
        return ysol

    @staticmethod
    def ode_multi(inner_cond, outer_cond, P, properties):
        """
        ode for the concentration distribution along the channel, two condensates
        :param inner_cond: temperature, molar fraction, and radius at inside;list
        :param outer_cond: temperature, molar fraction, and radius at outside; list
        :param P: pressure of mixture, bar
        :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
        :return: concentration and its slop
        """
        P = P * 1e5  # convert bar to pa
        [cp_c, cp_d, D_c, D_d, k] = properties
        [T1, x_c1, x_d1, r1] = inner_cond
        [T2, x_c2, x_d2, r2] = outer_cond
        D_ca, D_cb = 7.53e-6, 1.19e-5  # 9e-6, 1.5e-5  # 2.5e-5, 3e-5
        D_da, D_db = 1.15e-5, 1.62e-5  # 1.5e-5, 2.1e-5
        D_cm_id = 1 / (0.25 / D_ca + 0.75 / D_cb)
        D_dm_id = 1 / (0.25 / D_da + 0.75 / D_db)
        Nc_i = -(2 * P / R / (T2 + T1)) * D_cm_id * (x_c1 - x_c2) / (r1 - r2)
        Nd_i = -(2 * P / R / (T2 + T1)) * D_dm_id * (x_d1 - x_d2) / (r1 - r2)
        r = Nc_i / Nd_i

        def model(z, y):
            [xc, xd, Nd, T, dTdz] = y
            Nc = r * Nd
            D_cm = -(xc*(Nc+Nd)-Nc) / (0.25*Nc / D_ca + 0.75*Nc / D_cb)
            D_dm = -(xd*(Nc+Nd)-Nd) / (0.25*Nd / D_da + 0.75*Nd / D_db)
            dxc_dz = -Nc / D_cm / (P / R / T)
            dxd_dz = -Nd / D_dm / (P / R / T)

            # dNc_dz = -Nc / z
            dNd_dz = -Nd / z
            d2T_dz2 = -dTdz / z + dTdz * cp_c * Nc / k + dTdz * cp_d * Nd / k
            return np.vstack((dxc_dz, dxd_dz, dNd_dz, dTdz, d2T_dz2))

        def bound(ya, yb):
            return np.array([ya[0] - x_c1, ya[1] - x_d1, ya[3] - T1,
                             yb[0] - x_c2, yb[3] - T2])  # yb[1] - x_d2,

        xa, xb = r1, r2
        xini = np.linspace(xa, xb, 200)
        yini = np.zeros((5, xini.size))
        yini[0] = np.linspace(x_c1, x_c2, xini.size)
        yini[1] = np.linspace(x_d1, x_d2, xini.size)
        # yini[2] = -(2 * P / R / (T2 + T1)) * D_cm * (x_c1 - x_c2) / (r1 - r2)
        yini[2] = -(2 * P / R / (T2 + T1)) * D_dm_id * (x_d1 - x_d2) / (r1 - r2)
        yini[3] = np.linspace(T1, T2, xini.size)
        yini[4] = (T1 - T2) / (r1 - r2)
        res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-8, max_nodes=5000)
        xsol = np.linspace(xa, xb, 200)
        ysol = res.sol(xsol)
        ysol = np.insert(ysol, 2, ysol[2] * r, axis=0)
        return ysol

    def flux(self, Th, P, F_dict):
        """
        calculate the diffusional flux
        :param Th: temperature of gas in the reactor, K
        :param P: pressure of gas in the reactor, bar
        :param F_dict: gas component in the reactor, mol/s; ndarray
        :return: molar flux, heat flux, temperature variation and deviation of sim per length; dict
        """
        # calculate the correction to volumetric flow rate (m3/s)
        # calculate the partial pressure
        Ft = np.sum(F_dict)
        v = self.v0 * (self.P0 / P) * (Th / self.T0) * (Ft / self.Ft0)
        Pi = F_dict * R * Th / v * 1e-5  # bar

        # insulator parameter
        radium = [self.Din / 2, self.Do / 2]

        # calculate the partial pressure
        Pi_h = pd.Series(Pi, index=self.comp_list, dtype="float")  # pressure of gases in the reactor, bar
        xi_h = Pi_h / P
        if xi_h["Methanol"] < 3e-3:
            # if there is no reacted gas, end the calculation
            res = {
                "mflux": np.zeros(len(self.comp_list)),
                "hflux": 0,
                "Tvar": 0,
                "dev": 0
            }
            return res

        # vle calculation, determine the dew pressure
        vle = VLE(self.Tc, comp=xi_h)
        # print(vle.dew_p_all, vle.dew_p([2,3])['P']/(xi_h["Methanol"] + xi_h['H2O']))
        P_dew = vle.dew_p_all['P']
        # print(P_dew) vle.dew_p([2,3])['P']/(xi_h["Methanol"] + xi_h['H2O']) if vle.dew_p_all is None else
        if P < P_dew:
            qcv_delta = 1e5
            Tw = Th - 0.1
            while qcv_delta > 20:
                # condensation does not occur
                # gas properties inside the insulator
                mix_pro_ave = self.mixture_property((self.Tc + Tw) / 2, Pi_h)
                k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator

                # heat conduction along the insulator
                qcv_cond = -2 * np.pi * k_e * (self.Tc - Tw) / np.log(radium[1 - self.location] / radium[self.location])
                # heat convection inside the reactor
                qcv_conv = -self.convection(Th, P, F_dict) * radium[self.location] * 2 * np.pi * (Tw - Th)
                qcv_delta = 5 #abs(qcv_cond - qcv_conv)
                Tw -= 1
            # temperature variation inside the reactor
            property_h = self.mixture_property(Th, Pi_h)
            dT = qcv_cond / Ft / property_h["cp_m"]
            dT = -dT if self.location == 0 else dT
            res = {
                "mflux": np.zeros(len(self.comp_list)),
                "hflux": qcv_cond,
                "Tvar": dT,
                "dev": 0
            }
        else:
            # condensation occurs
            # calculate the composition of vapor and liquid phase
            flash_comp = vle.flash(P)
            xi_c = flash_comp.loc['V']
            Pi_c = xi_c * P

            qcv_delta = 1e5
            Tw = Th - 0.1
            while qcv_delta > 20:
                # gas properties inside the insulator
                property_c = self.mixture_property(self.Tc, Pi_c)
                property_w = self.mixture_property(Tw, Pi_h)
                mix_pro_ave = (property_w + property_c) / 2
                k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator

                # calculate the diffusional flux inside the insulator
                cold_cond = [self.Tc, xi_c["Methanol"], xi_c["H2O"], radium[1 - self.location]]
                hot_cond = [Th, xi_h["Methanol"], xi_h["H2O"], radium[self.location]]
                cond_list = [hot_cond, cold_cond]
                cal_property = [mix_pro_ave["cp_Methanol"], mix_pro_ave["cp_H2O"], 4.5e-5, 1.4e-5, k_e]
                # [xc, xd, Nc,Nd, T, dTdz]
                ode_res = self.ode_multi(cond_list[self.location], cond_list[self.location - 1], P, cal_property)

                # mass flux inside the insulator, mol/(s m)
                na_H20 = ode_res[3][-self.location] * radium[-self.location] * 2 * np.pi * vof
                na_CH3OH = ode_res[2][-self.location] * radium[-self.location] * 2 * np.pi * vof

                # heat conduction inside the insulator, W/m
                qcv_cond = -k_e * ode_res[5][-self.location] * radium[-self.location] * 2 * np.pi
                # heat convection inside the reactor
                qcv_conv = -self.convection(Th, P, F_dict) * radium[self.location] * 2 * np.pi * (Tw - Th)
                qcv_delta = 5 # abs(qcv_cond - qcv_conv)
                Tw -= 1
                # print(Th, Tw, qcv_cond, qcv_conv)
            gap_min = ode_res[1][-1] - cond_list[self.location - 1][2]

            property_h = self.mixture_property(Th, Pi_h)
            dT = qcv_cond / Ft / property_h["cp_m"]  # k/m
            dev = gap_min / xi_h["H2O"]
            if dev > 0.1: print([na_CH3OH, na_H20], dev, 'dev too big')

            dF = np.zeros_like(F_dict)
            dF[2:4] = [na_CH3OH, na_H20]
            if self.location == 0:
                dF = -1 * dF
                dT = -1 * dT
                if na_H20 < 0 or na_CH3OH < 0:
                    print('err')
                    print(cond_list, cal_property)
                    print("*" * 10)
            else:
                if na_H20 > 0 or na_CH3OH > 0:
                    print('err')
                    print(cond_list, cal_property)
                    print("*" * 10)
            res = {
                "mflux": dF,
                "hflux": qcv_cond,
                "Tvar": dT,
                "dev": dev
            }
        return res  # dF, dT * h_phi, dev


# P = np.array([0.25, 0.75]) * 50
# comp = ['CO2', 'H2']
# P = pd.Series(P, index=comp)
# print(Insulation.mixture_property(480, P))


# def ode_single(inner_cond, outer_cond, P, properties):
#     """
#     ode for the concentration distribution along the channel, only one condensate
#     :param inner_cond: temperature, molar fraction, and radius at inside;list
#     :param outer_cond: temperature, molar fraction, and radius at outside; list
# [cp, D, k] = properties

a = [373, 0.1, 0.03]
b = [313, 0.048, 0.04]

# def ode_multi(inner_cond, outer_cond, P, properties):
#     """
#     ode for the concentration distribution along the channel, two condensates
#     :param inner_cond: temperature, molar fraction, and radius at inside;list
#     :param outer_cond: temperature, molar fraction, and radius at outside; list
#     :param P: pressure of mixture, bar
#     :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
#     :return: concentration and its slop
#     """

# cold_cond = [333, 0.03, 0.01, 0.0225]
# hot_cond = [523, 0.08, 0.1, 0.015]
# cond_list = [hot_cond, cold_cond]
# cal_property = [60, 36, 4.5e-5, 1.4e-5, 0.2]
#
# y = Insulation.ode_multi(hot_cond,cold_cond,70,cal_property)
# print(y.shape)
# pro = [28, 2e-5, 30e-3]
# y = Insulation.ode_single(a, b, 1, pro)
# x = np.linspace(0.03, 0.04, 200)
# # [x, N, T, dTdz] = y
# # plt.plot(x, y[0,:])
# # plt.show()
#
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(x, y[0, :])
# # ax[0,0].title("x_H2O")
#
# ax[0, 1].plot(x, y[1, :])
# # ax[0,1].title("N_H2O")
# ax[1, 0].plot(x, y[2, :])
# # ax[1,0].title("T")
# ax[1, 1].plot(x, y[3, :])
# # ax[1,1].title("dTdz")
# plt.show()
