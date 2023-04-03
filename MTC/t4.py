import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from CoolProp.CoolProp import PropsSI

R = 8.314


def ode2(inner_cond, outer_cond, P, properties, r):
    """
    ode for the concentration distribution along the channel
    radial distribution
    the origin is located at the center of the circle
    :param inner_cond: temperature, molar fraction, and radius at inside;list
    :param outer_cond: temperature, molar fraction, and radius at outside; list
    :param P: pressure of mixture
    :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
    :return: concentration and its slop
    """
    [cp_c, cp_d, D_c, D_d, k] = properties
    [T1, r1, x_c1, x_d1] = inner_cond
    [T2, r2, x_c2, x_d2] = outer_cond
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
    yini[0] = np.linspace(x_c1, x_c2, xini.size)  # initial[0][0] #
    yini[1] = np.linspace(x_d1, x_d2, xini.size)  # initial[0][1]  #
    yini[2] = -0.01  # initial[1]  # [1]  # e-4
    yini[3] = np.linspace(T1, T2, xini.size)
    yini[4] = (T1 - T2) / (r1 - r2)
    res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-8, max_nodes=1000)
    xsol = np.linspace(xa, xb, 200)
    ysol = res.sol(xsol)
    return ysol


def ode4(inner_cond, outer_cond, P, properties, r):
    """
    ode for the concentration distribution along the channel
    radial distribution
    the origin is located at the center of the circle
    :param inner_cond: temperature, molar fraction, and radius at inside;list
    :param outer_cond: temperature, molar fraction, and radius at outside; list
    :param P: pressure of mixture
    :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
    :param r: r = Nc/Nd
    :return: concentration and its slop
    """
    [cp_c, cp_d, D_c, D_d, k] = properties
    [T1, r1, x_c1, x_d1] = inner_cond
    [T2, r2, x_c2, x_d2] = outer_cond
    D = (D_c + D_d) / 2

    # function

    def model(z, y):
        [xc, xd, Nd, T, dTdz] = y
        Nc = r * Nd
        dxc_dz = (-Nc * (1 - xc) + Nd * xc) / (P / R / T) / D
        dxd_dz = (-Nd * (1 - xd) + Nc * xd) / (P / R / T) / D
        dNd_dz = -Nd / z
        d2T_dz2 = -dTdz / z + dTdz * cp_c * Nc / k + dTdz * cp_d * Nd / k
        return np.vstack((dxc_dz, dxd_dz, dNd_dz, dTdz, d2T_dz2))

    def bound(ya, yb):
        return np.array([ya[0] - x_c1, ya[1] - x_d1, ya[3] - T1,
                         yb[0] - x_c2, yb[3] - T2])

    xa, xb = r1, r2
    xini = np.linspace(xa, xb, 200)
    yini = np.zeros((5, xini.size))
    yini[0] = np.linspace(x_c1, x_c2, xini.size)  # initial[0][0] #
    yini[1] = np.linspace(x_d1, x_d2, xini.size)  # initial[0][1]  #
    # yini[2] = initial[1][0]  # e-4
    yini[2] = -0.01  # initial[1]  # [1]  # e-4
    yini[3] = np.linspace(T1, T2, xini.size)
    yini[4] = (T1 - T2) / (r1 - r2)
    res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-8, max_nodes=5000)
    xsol = np.linspace(xa, xb, 200)
    ysol = res.sol(xsol)
    return ysol


def cold_comp(P, xi_in, psat_out):
    xi_ncon = np.delete(xi_in, [2, 3])
    xi_ncon = xi_ncon / np.sum(xi_ncon)
    pi_ncon_out = (P - np.sum(psat_out)) * xi_ncon
    pi_out = np.insert(pi_ncon_out, -1, psat_out)
    xi_out = pi_out / P
    return xi_out


def cond(T, xi, dx=0.02):
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
    pi_v = Pv_sat * np.array([1 - xg_H2O_sel, xg_H2O_sel])
    return pi_v
    # print(P_sat / 1e6)


# cold_gas = np.array([0.235517113, 0.745172371, 0.018675204, 0.008521366, 0.019310516])  # 5 MPa 1:2 353 K
hot_gas = np.array([0.209023659, 0.661347507, 0.060604809, 0.051885759, 0.017138265])

Tc = 353
gas_property = [275, 46, 4.5e-5, 1.8e-5, 0.3]
out_cond = [483, 0.045, hot_gas[2], hot_gas[3]]


def diffusion(dr=0.05):
    diff_min = 1e5
    for r_CH3OH_H20 in np.arange(0.05, 3, dr):
        pic_cond_v = cond(Tc, [r_CH3OH_H20 / (1 + r_CH3OH_H20), 1 / (1 + r_CH3OH_H20)])
        xic_v = cold_comp(40e5, hot_gas, pic_cond_v)
        in_cond = [353, 0.025, xic_v[2], xic_v[3]]

        res = ode2(in_cond, out_cond, 40e5, gas_property, r_CH3OH_H20)

        diff_xd = abs(res[1][-1] - hot_gas[3])
        if diff_xd < diff_min:
            diff_min = diff_xd
            r_sel = r_CH3OH_H20
    return r_sel


import pandas as pd

# lenth = len(np.arange(0.01, 0.1, 0.01))
temp_list = np.arange(0.01, 0.1, 0.01).tolist()
index_name = list(map(lambda x: str(round(x,2))+'_dr',temp_list))
col_name = list(map(lambda x: str(round(x,2))+'_dx',temp_list))
performance_error = pd.DataFrame(index=index_name, columns=col_name)
performance_time = pd.DataFrame(index=index_name, columns=col_name)
i = 0
for dr in np.arange(0.01, 0.1, 0.01):
    j = 0
    for dx in np.arange(0.01, 0.1, 0.01):
        a = time.time()
        r_CH3OH_H20 = diffusion(dr)
        pic_cond_v = cond(Tc, [r_CH3OH_H20 / (1 + r_CH3OH_H20), 1 / (1 + r_CH3OH_H20)])
        xic_v = cold_comp(40e5, hot_gas, pic_cond_v)
        in_cond = [353, 0.025, xic_v[2], xic_v[3]]
        res = ode2(in_cond, out_cond, 40e5, gas_property, r_CH3OH_H20)

        deviation = abs(res[1][-1] - hot_gas[3])
        b = time.time()
        time_cost = b - a
        performance_error.iloc[i, j] = deviation
        performance_time.iloc[i, j] = time_cost
        j += 1
    i += 1

performance_error.to_excel('er.xlsx')
performance_time.to_excel('time.xlsx')
# xsol = np.linspace(0.25, 0.45, 200)
# fig, axe = plt.subplots(2, 1)
# axe[0].plot(xsol, res[0])
# axe[0].plot(xsol, res[1])
# axe[0].legend(["CH3OH", "H2O"])
# axe[1].plot(xsol, res[3])
# plt.show()
