import matplotlib.pyplot as plt

from reactor import vof, ks
import numpy as np
import pandas as pd
import scipy
from CoolProp.CoolProp import PropsSI
from prop_calculator import VLE, mixture_property

R = 8.314


class Insulation:
    def __init__(self, Do, Din, n, location):

        # insulator parameters
        self.nit = n  # tube number of the insulator
        self.location = location
        self.Do, self.Din = Do, Din
        self.thick = (self.Do - self.Din) / 2
        self.comp_list = ["CO2", "H2", "Methanol", "H2O", "CO"]

    @staticmethod
    def bi_diff(T, P):
        k = 1.38e-23
        mass = np.array([44, 2, 32, 18, 28])
        sigma = np.array([3.941, 2.827, 3.626, 2.641, 3.690])
        epsilon = np.array([195.2, 59.7, 481.8, 809.1, 91.7])
        sigma_mix, epsilon_mix = np.zeros((5, 5)), np.zeros((5, 5))
        mass_mix, D_bi = np.zeros((5, 5)), np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                sigma_mix[i, j] = (sigma[i] + sigma[j]) / 2
                epsilon_mix[i, j] = (epsilon[i] * epsilon[j]) ** 0.5
                mass_mix[i, j] = 2 / (1 / mass[i] + 1 / mass[j])
                T_star = k * T / (epsilon_mix[i, j] * k)
                omega = 1.06036 / T_star ** 0.1561 + 0.193 / np.exp(0.47635 * T_star) \
                        + 1.03587 / np.exp(1.52996 * T_star) + 1.76474 / np.exp(3.89411 * T_star)
                D_bi[i, j] = 1e-4 * 0.00266 * T ** 1.5 / (P * mass_mix[i, j] ** 0.5 * sigma_mix[i, j] ** 2 * omega)
        return D_bi

    @staticmethod
    def bi_md(T, P):
        Dij_523 = np.array([[4.36E-07, 5.81E-06],
                            [5.80E-07, 5.74E-06]])
        # Dij_523 = np.array([[4.25E-07, 5.81E-06],
        #                     [5.62E-7, 5.74E-06]])
        return Dij_523 * (T / 523) ** 1.5 * (70 / P)

    def ode_multi(self, x_in, inner_cond, outer_cond, P, properties):
        """
        ode for the concentration distribution along the channel, two condensates
        :param x_in: molar fraction at input; pd
        :param inner_cond: temperature, molar fraction, and radius at inside;list
        :param outer_cond: temperature, molar fraction, and radius at outside; list
        :param P: pressure of mixture, bar
        :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
        :return: concentration and its slop
        """

        P = P * 1e5  # convert bar to pa
        x_main = x_in[["CO2", "H2"]] / np.sum(x_in[["CO2", "H2"]])
        [cp_c, cp_d, k] = properties
        [T1, x_c1, x_d1, r1] = inner_cond
        [T2, x_c2, x_d2, r2] = outer_cond

        D_bi = self.bi_md((T1 + T2) / 2, P / 1e5)

        D_31, D_32 = D_bi[0, 0], D_bi[0, 1]
        D_41, D_42 = D_bi[1, 0], D_bi[1, 1]

        D_cm_id = 1 / (0.25 / D_31 + 0.75 / D_32)
        D_dm_id = 1 / (0.25 / D_41 + 0.75 / D_42)
        Nc_i = -(2 * P / R / (T2 + T1)) * D_cm_id * (x_c1 - x_c2) / (r1 - r2)
        Nd_i = -(2 * P / R / (T2 + T1)) * D_dm_id * (x_d1 - x_d2) / (r1 - r2)
        r = Nc_i / Nd_i

        def model(z, y):
            [xc, xd, Nd, T, dTdz] = y
            Nc = r * Nd
            D_cm = -(xc * (Nc + Nd) - Nc) / (x_main["CO2"] * Nc / D_31 + x_main["H2"] * Nc / D_32)
            D_dm = -(xd * (Nc + Nd) - Nd) / (x_main["CO2"] * Nd / D_41 + x_main["H2"] * Nd / D_42)
            dxc_dz = -(Nc - xc * (Nc + Nd)) / (D_cm * (P / R / T))
            dxd_dz = -(Nd - xd * (Nc + Nd)) / (D_dm * (P / R / T))

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

    def flux(self, Th, P, F_dict, Tc):
        """
        calculate the diffusional flux
        :param Tc: temperature of water in the cold side of reactor, K
        :param Th: temperature of gas in the reactor, K
        :param P: pressure of gas in the reactor, bar
        :param F_dict: gas component in the reactor, mol/s; ndarray
        :return: molar flux, heat flux, temperature variation and deviation of sim per length; dict
        """
        # calculate the correction to volumetric flow rate (m3/s)
        # calculate the partial pressure

        Ft = np.sum(F_dict)

        # insulator parameter
        radium = [self.Din / 2, self.Do / 2]
        # calculate the molar fraction of the mix
        xi_h = pd.Series(F_dict / Ft, index=self.comp_list)
        if xi_h["Methanol"] < 3e-3 or xi_h['H2O'] < 1e-3:
            # if there is no reacted gas, end the calculation
            res = {
                "mflux": np.zeros(len(self.comp_list)),
                "hflux": 0, "Tvar": 0, "dev": 0
            }
            return res

        # vle calculation, determine the dew pressure
        vle_c = VLE(Tc, comp=self.comp_list)
        # P_dew_cds = vle.dew_p(y=xi_h['Methanol', 'H2O'])
        # P_dew = vle.dew_p(y=xi_h,x_guess=)
        # print(vle.dew_p_all, vle.dew_p([2,3])['P']/(xi_h["Methanol"] + xi_h['H2O']))
        # P_dew = vle.dew_p([2, 3])['P'] / (xi_h["Methanol"] + xi_h['H2O']) * 1.1  # vle.dew_p_all['P']
        P_sat_CH3OH = PropsSI('P', 'T', Tc, 'Q', 1, "Methanol")
        P_sat_H2O = PropsSI('P', 'T', Tc, 'Q', 1, "H2O")
        P_dew = (1 / (xi_h["Methanol"] / P_sat_CH3OH + xi_h['H2O'] / P_sat_H2O)) * 1e-5

        if P < P_dew:
            qcv_delta = 1e5
            Tw = Th - 0.1
            while qcv_delta > 20:
                # condensation does not occur
                # gas properties inside the insulator
                mix_pro_ave = mixture_property((Tc + Tw) / 2, xi_h, P)  # rho is not used, hence z=1 is used
                k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator

                # heat conduction along the insulator
                qcv_cond = -2 * np.pi * k_e * (Tc - Tw) / np.log(radium[1 - self.location] / radium[self.location])
                # heat convection inside the reactor
                # qcv_conv = -self.convection(Th, P, F_dict) * radium[self.location] * 2 * np.pi * (Tw - Th)
                qcv_delta = 5  # abs(qcv_cond - qcv_conv)
                Tw -= 1
            # temperature variation inside the reactor
            property_h = mixture_property(Th, xi_h, P)  # rho is not used, hence z=1 is used
            dT = qcv_cond / Ft / property_h["cp_m"]
            dT = -dT if self.location == 0 else dT
            res = {
                "mflux": np.zeros(len(self.comp_list)),
                "hflux": qcv_cond, "Tvar": dT, "dev": 0
            }
        else:
            # condensation occurs
            # calculate the composition of vapor and liquid phase
            flash_comp = vle_c.flash(P=P, mix=xi_h)
            xi_c = flash_comp.loc['V']

            qcv_delta = 1e5
            Tw = Th - 0.1
            while qcv_delta > 20:
                # gas properties inside the insulator
                property_c = mixture_property(Tc, xi_c, P)
                property_w = mixture_property(Tw, xi_h, P)
                mix_pro_ave = (property_w + property_c) / 2
                k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator
                # calculate the diffusional flux inside the insulator
                cold_cond = [Tc, xi_c["Methanol"], xi_c["H2O"], radium[1 - self.location]]
                hot_cond = [Th, xi_h["Methanol"], xi_h["H2O"], radium[self.location]]
                cond_list = [hot_cond, cold_cond]
                cal_property = [mix_pro_ave["cp_Methanol"], mix_pro_ave["cp_H2O"], k_e]
                # [xc, xd, Nc,Nd, T, dTdz]
                ode_res = self.ode_multi(xi_h, cond_list[self.location], cond_list[self.location - 1], P, cal_property)

                # mass flux inside the insulator, mol/(s m)
                # print(ode_res[3][-self.location] * vof / ((xi_h["Methanol"] - xi_c["Methanol"]) * self.P0 * 1e5))
                # print(ode_res[2][-self.location] * vof / ((xi_h["H2O"] - xi_c["H2O"]) * self.P0 * 1e5))
                na_H20 = ode_res[3][-self.location] * radium[-self.location] * 2 * np.pi * vof
                na_CH3OH = ode_res[2][-self.location] * radium[-self.location] * 2 * np.pi * vof

                # heat conduction inside the insulator, W/m
                qcv_cond = -k_e * ode_res[5][-self.location] * radium[-self.location] * 2 * np.pi
                # heat convection inside the reactor
                # qcv_conv = -self.convection(Th, P, F_dict) * radium[self.location] * 2 * np.pi * (Tw - Th)
                qcv_delta = 5  # abs(qcv_cond - qcv_conv)
                Tw -= 1
                # print(Th, Tw, qcv_cond, qcv_conv)
            gap_min = ode_res[1][-1] - cond_list[self.location - 1][2]
            # [xc, xd, Nc, Nd, T, dTdz]
            # xsol = np.linspace(0.02, 0.03, 200)
            # plt.plot(xsol, ode_res[1])
            # plt.show()
            # plt.plot(xsol, ode_res[3])
            # plt.show()

            property_h = mixture_property(Th, xi_h, P)
            dT = qcv_cond / Ft / property_h["cp_m"]  # k/m
            dev = gap_min / xi_h["H2O"]
            if dev > 0.1: print([na_CH3OH, na_H20], dev, 'dev too big')

            dF = np.zeros_like(F_dict)
            dF[2:4] = [na_CH3OH, na_H20]
            # print(k_e)
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
# print(Insulation.bi_diff(298, 1))

# def ode_single(inner_cond, outer_cond, P, properties):
#     """
#     ode for the concentration distribution along the channel, only one condensate
#     :param inner_cond: temperature, molar fraction, and radius at inside;list
#     :param outer_cond: temperature, molar fraction, and radius at outside; list
# [cp, D, k] = properties

# a = [373, 0.1, 0.03]
# b = [313, 0.048, 0.04]

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
