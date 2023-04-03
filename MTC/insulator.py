import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
from CoolProp.CoolProp import PropsSI


def ode_bvp(inner_cond, outer_cond, P, properties):
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
    [cp, D, k] = properties
    [T1, c1, r1] = inner_cond
    [T2, c2, r2] = outer_cond

    # function

    def model(z, y):
        x, T = y[0], y[2]
        dxdz, dTdz = y[1], y[3]
        d2x_dz2 = -dxdz / z + dTdz / T * dxdz - dxdz ** 2 / (1 - x)
        d2T_dz2 = -dTdz / z - cp * P * D / k / R / (1 - x) * dxdz / T * dTdz
        return np.vstack((dxdz, d2x_dz2, dTdz, d2T_dz2))

    def bound(ya, yb):
        return np.array([ya[0] - c1, ya[2] - T1, yb[0] - c2, yb[2] - T2])

    xa, xb = r1, r2
    xini = np.linspace(xa, xb, 11)
    yini = np.zeros((4, xini.size))
    yini[0] = np.linspace(c1, c2, xini.size)
    yini[1] = (c1 - c2) / (r1 - r2)
    yini[2] = np.linspace(T1, T2, xini.size)
    yini[3] = (T1 - T2) / (r1 - r2)
    res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-10, max_nodes=1000)
    xsol = np.linspace(xa, xb, 500)
    ysol = res.sol(xsol)
    return ysol


def mixture_property(T, P, cond_gas):
    """
    calculate the properties of gas mixture
    :param T: gas temperature, K
    :param P: partial pressure, pa
    :return: thermal conductivity W/(m K), viscosity Pa s, heat capacity J/mol/K; pd.series
    """
    # prepare data for calculation
    n = len(P.index)  # number of gas species
    [cp, k, vis, M] = np.empty((4, n))
    mol_fraction = (P / P.sum()).to_numpy()  # mol fraction of gases
    i = 0

    # calculate the properties of pure gases
    for comp in P.index:
        gas = "N2" if comp == "CO" else comp  # "CO" is not available in CoolProp
        # thermal conductivity, W/(m K)
        k[i] = PropsSI('L', 'T', T, 'P', P[comp] - 100, gas)
        # viscosity, Pa S
        vis[i] = PropsSI('V', 'T', T, 'P', P[comp] - 100, gas)
        # heat capacity, J/(mol K)
        cp[i] = PropsSI('CPMOLAR', 'T', T, 'P', P[comp] - 100, gas) \
            # if gas != cond_gas else PropsSI('CPMOLAR', 'T', T, 'Q', 1,gas)
        M[i] = PropsSI('MOLARMASS', 'T', T, 'P', 1e5, gas)  # molar weight, g/mol
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

    return pd.Series([k_m, vis_m, cp_m], index=["k", "vis", "cp"])


def mflux(Th, P, F_dict, insulator_data, cond_gas="H2O"):
    """
    calculate the diffusional flux
    :param Th: temperature of gas in the reactor, K
    :param P: pressure of gas in the reactor, bar
    :param F_dict: gas component in the reactor, pd.Series,
    :param insulator_data: insulator parameter
    :param cond_gas: condensate
    :return:
    """
    P *= 1e5  # convert bar to Pa
    Tc = insulator_data["Tc"]
    Ts = PropsSI("Tcrit",cond_gas)

    # calculate the partial pressure
    pi_h = pd.Series(index=F_dict.index, dtype="float")  # pressure of gases in the reactor, Pa
    pi_c = pd.Series(index=F_dict.index, dtype="float")  # pressure of gases in the condenser, Pa
    Ft0 = F_dict.sum()  # total molar flux, mol/s
    for comp in F_dict.index: pi_h[comp] = F_dict[comp] / Ft0 * P

    pi_c[cond_gas] = PropsSI('P', 'T', Tc, 'Q', 1, cond_gas)
    # to judge if the partial pressure of H2O are large enough
    if pi_h["Methanol"] < 1e-8:
        # if there is no reacted gas, end the calculation
        return 0, 0, 0
    if pi_c[cond_gas] > pi_h[cond_gas]:
        # if the partial pressure of condensate is low, molar ratio of stream won't change
        pi_c[cond_gas] = pi_h[cond_gas]

    # calculate the heat conductivity and the heat capacity
    for comp in F_dict.index:
        if comp != cond_gas: pi_c[comp] = (P - pi_c[cond_gas]) * (F_dict[comp] / (Ft0 - F_dict[cond_gas]))

    property_h, property_c = mixture_property(Th, pi_h, cond_gas), mixture_property(Tc, pi_c, cond_gas)
    k_v = (property_h["k"] + property_c["k"]) / 2  # heat conductivity of mixed gases, W/m/K
    cp_v = (property_h["cp"] + property_c["cp"]) / 2
    k_e = k_v * vof + ks * (1 - vof)  # effective heat conductivity of the insulator

    # calculate the diffusional flux
    c_in = P / 8.314 / Th  # mol/m3
    xc_mol_h = pi_h[cond_gas] / P
    xc_mol_c = pi_c[cond_gas] / P

    # read the insulator parameter
    location = insulator_data["io"]
    diameter = [insulator_data["Din"], insulator_data["Do"]]
    position = 0 if location == "in" else 1
    cond_list = [[Th, xc_mol_h] + [diameter[position]], [Tc, xc_mol_c] + [diameter[1 - position]]]
    xc_c_dis = ode_bvp(cond_list[position], cond_list[position - 1], P, [cp_v, 5e-5, k_v])
    na_mass_t = -1.8e-5 * xc_c_dis[1][-position] * c_in / (1 - xc_mol_h) * diameter[-position] * np.pi
    qcv = -k_e * xc_c_dis[3][-position] * diameter[-position] * np.pi

    dT = qcv / Ft0 / cp_v  # k/m
    return na_mass_t, dT, qcv


ks = 0.2  # W/m K
vof = 0.8
R = 8.314
#
# #
# property_gas = [14.535 * 2, 30e5, 1e-5, 0.2]  # J/mol K, Pa, m2/s, W/m K
# a = [0.016356019, 0.05279398, 0.002325577, 0.004188539, 0.001862962]
#
# a = [0.019942661, 0.060032052, 0.000499862, 0.000601897, 0.000102035]
# # # # # a = [0.065428371,0.211165743,0.009309545,0.06674986,0.007440315]
# # # # # a = [0.080776961, 0.242792585, 0.00117042, 0.00140127, 0.00023085]
# F = pd.Series(a, index=["CO2", "H2", "Methanol", "H2O", "CO"])
# # # a,d = mflux(487, 30, F, 343,"in")
# para = {"io": "in", "Tc": 343, "Dm": 0.05, "Dc": 0.09}
# print(mflux(487, 30, F, para))
# print(PropsSI('L', 'T', 423, 'Q', 1, "Methanol"))
# p = PropsSI('P', 'T', 423, 'Q', 1, "Methanol")
# print(PropsSI('L', 'T', 423, 'P', p-100, "Methanol"))
