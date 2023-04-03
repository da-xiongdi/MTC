import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from CoolProp.CoolProp import PropsSI
import pandas as pd

R = 8.314


def ode_multi(inner_cond, outer_cond, P, properties, r):
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
    yini[0] = np.linspace(x_c1, x_c2, xini.size)  # initial[0][0] #
    yini[1] = np.linspace(x_d1, x_d2, xini.size)  # initial[0][1]  #
    yini[2] = -0.01  # initial[1]  # [1]  # e-4
    yini[3] = np.linspace(T1, T2, xini.size)
    yini[4] = (T1 - T2) / (r1 - r2)
    res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-8, max_nodes=1000)
    xsol = np.linspace(xa, xb, 200)
    ysol = res.sol(xsol)
    return ysol


def cold_comp(Pi_in, psat_out):
    P_t = Pi_in.sum()
    xi_in = Pi_in / P_t
    xi_ncon = xi_in.drop(["Methanol", "H2O"])
    xi_ncon = xi_ncon / xi_ncon.sum()
    pi_ncon_out = (P_t - psat_out.sum()) * xi_ncon
    pi_out = pd.concat([pi_ncon_out, psat_out])
    pi_out = pi_out[Pi_in.index.tolist()]
    return pi_out


def p_sat(T, xi, dx=0.01):
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
    pi_v_pd = pd.Series(pi_v, index=["Methanol", "H2O"])
    return pi_v_pd

# print(PropsSI('D', 'T', 300, 'P', 101325, 'REFPROP::CO'))
