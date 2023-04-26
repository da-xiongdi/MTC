import matplotlib.pyplot as plt

from reactor import Reaction, vof, ks
import numpy as np
import pandas as pd
import scipy
from CoolProp.CoolProp import PropsSI

R = 8.314


class Insulation(Reaction):
    def __init__(self, reactor_para, chem_para, feed_para, insulator_para, r_CH3OH_H2O):
        super(Insulation, self).__init__(reactor_para, chem_para, feed_para)

        self.r_CH3OH_H2O = r_CH3OH_H2O  # molar ratio of liquid phase
        # insulator parameters
        self.insulator_para = insulator_para
        self.nit = self.insulator_para["nit"]  # tube number of the insulator
        self.Din = self.insulator_para['Din']
        self.Do = self.Din + self.insulator_para['Thick'] * 2
        self.Tc = self.insulator_para['Tc']
        self.location = self.insulator_para["io"]

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
    def mixture_property(T, Pi_gas, component):
        """
        calculate the properties of gas mixture
        :param T: gas temperature, K
        :param Pi_gas: partial pressure, bar; pd.Serize
        :param component: component of gas, list
        :return: thermal conductivity W/(m K), viscosity Pa s, heat capacity J/mol/K; pd.series
        """
        # prepare data for calculation
        n = len(Pi_gas.index)  # number of gas species

        [cp, k, vis, M] = np.empty((4, n))
        mol_fraction = Pi_gas.values / np.sum(Pi_gas.values)  # mol fraction of gases
        Pi_gas = Pi_gas * 1e5  # convert bar to pa

        i = 0
        # calculate the properties of pure gases
        for comp in component:
            gas = "N2" if comp == "CO" else comp  # "CO" is not available in CoolProp
            # thermal conductivity, W/(m K)
            if Pi_gas[comp] < 1000:
                k[i] = vis[i] = cp[i] = 1e-5
            else:
                k[i] = PropsSI('L', 'T', T, 'P', Pi_gas[comp] - 100, gas)
                # viscosity, Pa S
                vis[i] = PropsSI('V', 'T', T, 'P', Pi_gas[comp] - 100, gas)
                # heat capacity, J/(mol K)
                cp[i] = PropsSI('CPMOLAR', 'T', T, 'P', Pi_gas[comp] - 100, gas)
            # molar weight, g/mol
            M[i] = PropsSI('MOLARMASS', 'T', T, 'P', 1e5, gas)
            i += 1

        # calculate the properties of mixture
        cp_m = np.sum(cp * mol_fraction)
        phi, denominator = np.ones((n, n)), np.ones((n, n))  # Wilke coefficient
        vis_m, k_m = 0, 0
        for i in range(n):
            for j in np.arange(n):
                phi[i, j] = (1 + (vis[i] / vis[j]) ** 0.5 * (M[j] / M[i]) ** 0.25) ** 2 / (8 * (1 + M[i] / M[j])) ** 0.5
                denominator[i, j] = mol_fraction[j] * phi[i, j] if i != j else 0
            vis_m += mol_fraction[i] * vis[i] / np.sum(denominator[i])
            k_m += mol_fraction[i] * k[i] / np.sum(denominator[i])
        return pd.Series([k_m, vis_m, cp[2], cp[3], cp_m],
                         index=["k", "vis", 'cp_' + Pi_gas.index[2], 'cp_' + Pi_gas.index[3], "cp_m"])

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
        D_da, D_db = 1.1e-5, 1.5e-5
        D_cm = 1 / (0.25 / D_ca + 0.75 / D_cb)
        D_dm = 1 / (0.25 / D_da + 0.75 / D_db)
        [cp, D, k] = properties
        [T1, c1, r1] = inner_cond
        [T2, c2, r2] = outer_cond

        def model(z, y):
            [x, N, T, dTdz] = y
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
    def ode_multi2(inner_cond, outer_cond, P, properties, r):
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
        D_cd = 2.6e-5

        def model(z, y):
            [xc, xd, Nd, T, dTdz] = y
            Nc = r * Nd
            dxd_dz = (-Nd * ((1 - xc - xd) / D_d + xc / D_cd) + Nc * xd / D_cd) / (P / R / T)
            dxc_dz = (-Nc * ((1 - xc - xd) / D_c + xd / D_cd) + Nd * xc / D_cd) / (P / R / T)
            dNd_dz = -Nd / z
            d2T_dz2 = -dTdz / z + dTdz * cp_c * Nc / k + dTdz * cp_d * Nd / k
            return np.vstack((dxc_dz, dxd_dz, dNd_dz, dTdz, d2T_dz2))

        def bound(ya, yb):
            return np.array([ya[0] - x_c1, ya[1] - x_d1, ya[3] - T1,
                             yb[0] - x_c2, yb[3] - T2])

        xa, xb = r1, r2
        xini = np.linspace(xa, xb, 200)
        yini = np.zeros((5, xini.size))
        yini[0] = np.linspace(x_c1, x_c2, xini.size)
        yini[1] = np.linspace(x_d1, x_d2, xini.size)
        yini[2] = 0.001 if x_c1 > x_c2 else -0.001
        yini[3] = np.linspace(T1, T2, xini.size)
        yini[4] = (T1 - T2) / (r1 - r2)
        res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-8, max_nodes=1000)
        xsol = np.linspace(xa, xb, 200)
        ysol = res.sol(xsol)
        return ysol

    @staticmethod
    def ode_multi3(inner_cond, outer_cond, P, properties, r):
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
        D_cd = 2.6e-5

        def model(z, y):
            [xc, xd, Nc, Nd, T, dTdz] = y
            dxd_dz = (-Nd * ((1 - xc - xd) / D_d + xc / D_cd) + Nc * xd / D_cd) / (P / R / T)
            dxc_dz = (-Nc * ((1 - xc - xd) / D_c + xd / D_cd) + Nd * xc / D_cd) / (P / R / T)
            dNd_dz = -Nd / z
            dNc_dz = -Nc / z
            d2T_dz2 = -dTdz / z + dTdz * cp_c * Nc / k + dTdz * cp_d * Nd / k
            return np.vstack((dxc_dz, dxd_dz, dNc_dz, dNd_dz, dTdz, d2T_dz2))

        def bound(ya, yb):
            return np.array([ya[0] - x_c1, ya[1] - x_d1, ya[3] - T1,
                             yb[0] - x_c2, yb[1] - x_d2, yb[3] - T2])

        xa, xb = r1, r2
        xini = np.linspace(xa, xb, 200)
        yini = np.zeros((6, xini.size))
        yini[0] = np.linspace(x_c1, x_c2, xini.size)
        yini[1] = np.linspace(x_d1, x_d2, xini.size)
        yini[2] = 0.001 if x_c1 > x_c2 else -0.001
        yini[4] = np.linspace(T1, T2, xini.size)
        yini[5] = (T1 - T2) / (r1 - r2)
        res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-8, max_nodes=1000)
        xsol = np.linspace(xa, xb, 200)
        ysol = res.sol(xsol)

        return ysol

    @staticmethod
    def ode_multi(inner_cond, outer_cond, P, properties, r):
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
        D_cm = 1 / (0.25 / D_ca + 0.75 / D_cb)
        D_dm = 1 / (0.25 / D_da + 0.75 / D_db)
        Nc_i = -(2 * P / R / (T2 + T1)) * D_cm * (x_c1 - x_c2) / (r1 - r2)
        Nd_i = -(2 * P / R / (T2 + T1)) * D_dm * (x_d1 - x_d2) / (r1 - r2)
        r = Nc_i / Nd_i

        def model(z, y):
            [xc, xd, Nd, T, dTdz] = y
            # D_cm = -(xc*(Nc+Nd)-Nc) / (0.25*Nc / D_ca + 0.75*Nc / D_cb)
            # D_dm = -(xd*(Nc+Nd)-Nd) / (0.25*Nd / D_da + 0.75*Nd / D_db)
            Nc = r * Nd
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
        yini[2] = -(2 * P / R / (T2 + T1)) * D_dm * (x_d1 - x_d2) / (r1 - r2)
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
        :return:
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
        if Pi_h["Methanol"] < 1e-5:
            # if there is no reacted gas, end the calculation
            return np.zeros(len(self.comp_list)), 0, 0
        xi_h = Pi_h / P

        h_phi = 1
        # to judge if the partial pressure of condensate are large enough
        xi_condensate = xi_h["Methanol"] + xi_h["H2O"]
        mix_condensate = 'HEOS::Methanol[%s]&H2O[%s]' % (xi_h["Methanol"] / xi_condensate, xi_h["H2O"] / xi_condensate)
        Pv_sat = PropsSI('P', 'T', self.Tc, 'Q', 1, mix_condensate) * 1e-5
        if (Pi_h["Methanol"] + Pi_h["H2O"]) < Pv_sat:
            # if the partial pressure of condensate is low, only heat diffusion while no mass diffusion
            mix_pro_ave = self.mixture_property((self.Tc + Th) / 2, Pi_h, self.comp_list)
            property_h = self.mixture_property(Th, Pi_h, self.comp_list)
            k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator
            qcv = -2 * np.pi * k_e * (self.Tc - Th) / np.log(radium[1 - self.location] / radium[self.location])
            dT = qcv / Ft / property_h["cp_m"]
            dT = -dT if self.location == 0 else dT
            return np.zeros(len(self.comp_list)), dT * h_phi, 0

        # to determine the molar flux of condensate
        # for r_CH3OH_H2O in np.arange(0.1, 1, 0.1):
        Pi_c_cond = self.p_sat(self.Tc, [self.r_CH3OH_H2O / (1 + self.r_CH3OH_H2O), 1 / (1 + self.r_CH3OH_H2O)])
        Pi_c = pd.Series(self.cold_comp(Pi_h.values, Pi_c_cond), index=self.comp_list)
        xi_c = Pi_c / P

        # calculate the heat conductivity and the heat capacity
        property_h = self.mixture_property(Th, Pi_h, self.comp_list)
        property_c = self.mixture_property(self.Tc, Pi_c, self.comp_list)
        mix_pro_ave = (property_h + property_c) / 2
        k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator
        dev = 0
        if xi_c["Methanol"] > xi_h["Methanol"] * 0.99:
            # only H2O diffused
            cold_cond = [self.Tc, xi_c["H2O"], radium[1 - self.location]]
            hot_cond = [Th, xi_h["H2O"], radium[self.location]]
            cond_list = [hot_cond, cold_cond]
            cal_property = [mix_pro_ave["cp_H2O"], 1.4e-5, k_e]
            res = self.ode_single(cond_list[self.location], cond_list[self.location - 1], P, cal_property)
            na_H20, na_CH3OH = res[1][-self.location] * radium[-self.location] * 2 * np.pi * vof, 0  # mol/(s m)
            qcv = -k_e * res[3][-self.location] * radium[-self.location] * 2 * np.pi
            # print(Th)
            # print(qcv, k_e, Ft, property_h['cp_m'])
            dT = qcv / Ft / property_h["cp_m"]  # k/m
        elif xi_c["H2O"] > xi_h["H2O"] * 0.99:
            # only CH3OH diffuse
            cold_cond = [self.Tc, xi_c["Methanol"], radium[1 - self.location]]
            hot_cond = [Th, xi_h["Methanol"], radium[self.location]]
            cond_list = [hot_cond, cold_cond]
            cal_property = [mix_pro_ave["cp_Methanol"], 4.5e-5, k_e]
            res = self.ode_single(cond_list[self.location], cond_list[self.location - 1], P, cal_property)
            na_H20, na_CH3OH = 0, res[1][-self.location] * radium[-self.location] * 2 * np.pi * vof  # mol/(s m)
            qcv = -k_e * res[3][-self.location] * radium[-self.location] * 2 * np.pi

            dT = qcv / Ft / property_h["cp_m"]  # k/m
        else:
            # guess a ratio between N_CH3OH and N_H2O
            # determine the saturated pressure in cold side
            # perform the calculation of diffusional flux
            # the best ratio is selected by comparing the xi_h["H2O"]
            cold_cond = [self.Tc, xi_c["Methanol"], xi_c["H2O"], radium[1 - self.location]]
            hot_cond = [Th, xi_h["Methanol"], xi_h["H2O"], radium[self.location]]
            cond_list = [hot_cond, cold_cond]
            cal_property = [mix_pro_ave["cp_Methanol"], mix_pro_ave["cp_H2O"], 4.5e-5, 1.4e-5, k_e]
            # print(cold_cond, hot_cond, cal_property)
            res = self.ode_multi(cond_list[self.location], cond_list[self.location - 1], P, cal_property, 0)
            # /m * m2/s * mol/m3 = mol/s/m2
            na_H20 = res[3][-self.location] * radium[-self.location] * 2 * np.pi * vof  # mol/(s m)
            na_CH3OH = res[2][-self.location] * radium[-self.location] * 2 * np.pi * vof
            # calculate the heat flux
            qcv = -k_e * res[5][-self.location] * radium[-self.location] * 2 * np.pi
            gap_min = res[1][-1] - cond_list[self.location - 1][2]

            dT = qcv / Ft / property_h["cp_m"]  # k/m
            dev = gap_min / xi_h["H2O"]
            if dev > 0.1: print([na_CH3OH, na_H20], dev, 'dev too big')
        # print(k_e / 0.01)
        # print(qcv)

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
        return dF, dT * h_phi, dev

# # pi, pj = 0.04181582, 0.332947903
# # xi, xj = pi / (pi + pj), pj / (pi + pj)
# # psat_i = PropsSI('P', 'T', 343, 'Q', 1, 'Methanol') * 1e-5
# # psat_j = PropsSI('P', 'T', 343, 'Q', 1, 'H2O') * 1e-5
# #
# # p = pi + pj
# # mix_condensate = 'HEOS::Methanol[%s]&H2O[%s]' % (xi, xj)
# # p_d = PropsSI('P', 'T', 343, 'Q', 1, mix_condensate) * 1e-5
# # p_b = PropsSI('P', 'T', 343, 'Q', 0, mix_condensate) * 1e-5
# # print(psat_i, psat_j, 1 / (xi / psat_i + xj / psat_j), xi * psat_i + xj * psat_j)
# # print(psat_i, psat_j, p_d, p_b)
# #
# # ki = psat_i / p
# # kj = psat_j / p
# #
# # print(ki, kj)
# #
# #
# # def f(V):
# #     return xi * ki / (1 + V * (ki - 1)) + xj * kj / (1 + V * (kj - 1))
# #
# #
# # res = scipy.optimize.fsolve(f, [1])
# # print(res)
#
# psat_i = PropsSI('P', 'T', 353, 'Q', 1, 'Methanol') * 1e-5
# psat_j = PropsSI('P', 'T', 353, 'Q', 1, 'H2O') * 1e-5
#
# x_i = (0.5*psat_j + 0.5 * psat_i)
#
# mix_condensate = 'HEOS::Methanol[%s]&H2O[%s]' % (xi, xj)
# p_d = PropsSI('P', 'T', 343, 'Q', 1, mix_condensate) * 1e-5
# p_b = PropsSI('P', 'T', 343, 'Q', 0, mix_condensate) * 1e-5

# [[529.8331065684047, 0.0172516965849795, 0.028182381875855166, 0.02], [333, 0.008126980082540864, 0.002566414762907641, 0.04]]
# [62.0569658571729, 35.371468278312946, 4.5e-05, 1.4e-05, 0.2948365879111804]
# [T1, c1, r1] = inner_cond
# [cp, D, k] = properties
# in_cond = [529.8331065684047, 0.0172516965849795, 0.028182381875855166, 0.02]
# out_cond = [333, 0.008126980082540864, 0.002566414762907641, 0.04]
# cal = [62.0569658571729, 35.371468278312946, 0.9e-05, 1.4e-05, 0.2948365879111804]
# res = Insulation.ode_multi(in_cond, out_cond, 50, cal, 1)
# res2 = Insulation.ode_single([529.8331065684047, 0.028182381875855166, 0.02], [333, 0.002566414762907641, 0.04], 50,
#                              [35.37, 1, 0.3])
# # [x, N, T, dTdz] = y
# # # print(res[0])
# xsol = np.linspace(0.02, 0.04, 200)
#
# # # [xc, xd, Nc, Nd, T, dTdz] = y [xc, xd, Nd, T, dTdz] = y
# print(res[0][0], res[0][-1])
# print(res[1][0], res[1][-1])
# # fig, axe = plt.subplots(2, 2)
# # axe[0][0].plot(xsol, res[0])
# # axe[0][0].plot(xsol, res[1])
# # axe[0][1].plot(xsol, res[2]*xsol)
# # axe[0][1].plot(xsol, res[3]*xsol)
# # axe[1][0].plot(xsol, res[4])
# # axe[1][1].plot(xsol, res[5])
# # plt.show()
#
# plt.plot(xsol, res[1])
# plt.plot(xsol, res2[0])
# plt.show()
#
# plt.plot(xsol, res[5])
# plt.plot(xsol, res2[3])
# plt.show()
