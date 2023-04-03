import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

import diffusion as diff
import json


def mixture_property(T, P):
    """
    calculate the properties of gas mixture
    :param T: gas temperature, K
    :param P: partial pressure, pa
    :return: thermal conductivity W/(m K), viscosity Pa s, heat capacity J/mol/K; pd.series
    """
    # prepare data for calculation
    n = len(P.index)  # number of gas species
    pure_pro = pd.DataFrame(index=["cp", "k", "vis", "M"], columns=[P.index])
    [cp, k, vis, M] = np.empty((4, n))
    mol_fraction = (P / P.sum()).to_numpy()  # mol fraction of gases
    # mol_fraction = P / P.sum()#).to_numpy()  # mol fraction of gases
    i = 0

    # calculate the properties of pure gases
    for comp in P.index:
        gas = "N2" if comp == "CO" else comp  # "CO" is not available in CoolProp
        # thermal conductivity, W/(m K)
        # pure_pro.loc['k', comp] = PropsSI('L', 'T', T, 'P', P[comp] - 100, gas)
        k[i] = PropsSI('L', 'T', T, 'P', P[comp] - 100, gas)
        # viscosity, Pa S
        # pure_pro.loc['vis', comp] = PropsSI('V', 'T', T, 'P', P[comp] - 100, gas)
        vis[i] = PropsSI('V', 'T', T, 'P', P[comp] - 100, gas)
        # heat capacity, J/(mol K)
        # pure_pro.loc['cp', comp] = PropsSI('CPMOLAR', 'T', T, 'P', P[comp] - 100, gas)
        cp[i] = PropsSI('CPMOLAR', 'T', T, 'P', P[comp] - 100, gas)  # \
        # if gas != cond_gas else PropsSI('CPMOLAR', 'T', T, 'Q', 1,gas)
        M[i] = PropsSI('MOLARMASS', 'T', T, 'P', 1e5, gas)  # molar weight, g/mol
        # pure_pro.loc['M', comp] = PropsSI('MOLARMASS', 'T', T, 'P', P[comp] - 100, gas)
        i += 1

    # calculate the properties of mixture
    cp_m = np.sum(cp * mol_fraction)
    # cp_m = (pure_pro.loc['cp'] * mol_fraction).sum()
    phi, denominator = np.ones((n, n)), np.ones((n, n))  # Wilke coefficient
    vis_m, k_m = 0, 0
    for i in range(n):
        for j in np.arange(n):
            phi[i, j] = (1 + (vis[i] / vis[j]) ** 0.5 * (M[j] / M[i]) ** 0.25) ** 2 / (8 * (1 + M[i] / M[j])) ** 0.5
            denominator[i, j] = mol_fraction[j] * phi[i, j] if i != j else 0
        vis_m += mol_fraction[i] * vis[i] / np.sum(denominator[i])
        k_m += mol_fraction[i] * k[i] / np.sum(denominator[i])

    return pd.Series([k_m, vis_m, cp[2], cp[3], cp_m],
                     index=["k", "vis", 'cp_' + P.index[2], 'cp_' + P.index[3], "cp_m"])


def flux(Th, P, F_dict, insulator_data):
    """
    calculate the diffusional flux
    :param Th: temperature of gas in the reactor, K
    :param P: pressure of gas in the reactor, bar
    :param F_dict: gas component in the reactor, pd.Series,
    :param insulator_data: insulator parameter
    :return:
    """
    P *= 1e5  # convert bar to Pa
    Tc = insulator_data["Tc"]
    cond_gases = ["Methanol", "H2O"]

    # read the insulator parameter
    location = insulator_data["io"]
    radium = [insulator_data["Din"] / 2, insulator_data["Do"] / 2]
    position = 0 if location == "in" else 1

    # calculate the partial pressure
    pi_h = pd.Series(index=F_dict.index, dtype="float")  # pressure of gases in the reactor, Pa
    Ft0 = F_dict.sum()  # total molar flux, mol/s
    for comp in F_dict.index: pi_h[comp] = F_dict[comp] / Ft0 * P
    if pi_h["Methanol"] < 1e-8:  # if there is no reacted gas, end the calculation
        return 0, 0, 0, 0
    xi_h = pi_h / P
    # to judge if the partial pressure of condensate are large enough
    pi_c_sat = {}
    for cond_gas in cond_gases:
        pi_c_sat[cond_gas] = PropsSI('P', 'T', Tc, 'Q', 1, cond_gas)
    if pi_c_sat["Methanol"] > pi_h["Methanol"] and pi_c_sat["H2O"] > pi_h["H2O"]:
        # if the partial pressure of condensate is low, molar ratio of stream won't change
        na_H20, na_CH3OH = 0, 0
        mix_pro_ave = mixture_property((Tc + Th) / 2, pi_h)
        k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator
        qcv = -2 * np.pi * k_e * (Tc - Th) / np.log(radium[1 - position] / radium[position])
        dT = qcv / Ft0 / mix_pro_ave["cp_m"]
        return na_H20, na_CH3OH, dT, 0

    # to determine the molar flux of condensate
    # guess a ratio between N_CH3OH and N_H2O
    # determine the saturated pressure in cold side
    # perform the calculation of diffusional flux
    # the best ratio is selected by comparing the xi_h["H2O"]
    gap_min = 1e5
    rmin, rmax = 0.1, 1.6
    r_guess = -629.59 * xi_h["H2O"] ** 2 + 64.735 * xi_h["H2O"] - 0.214  # x2 + 64.735x
    rmin = max(r_guess * 0.5, 0)
    rmax = min(r_guess * 1.5, 1.6)
    # for dr in np.linspace(0.5, 0.01, 10):
    for r_CH3OH_H20 in np.arange(rmin, rmax, 0.05):
        # print(r_CH3OH_H20)
        pi_c_cond = diff.p_sat(Tc, [r_CH3OH_H20 / (1 + r_CH3OH_H20), 1 / (1 + r_CH3OH_H20)])
        pi_c = diff.cold_comp(pi_h, pi_c_cond)
        xi_c = pi_c / P
        cold_cond = [Tc, xi_c["Methanol"], xi_c["H2O"]]
        hot_cond = [Th, xi_h["Methanol"], xi_h["H2O"]]

        hot_cond.append(radium[position]), cold_cond.append(radium[1 - position])
        cond_list = [hot_cond, cold_cond]

        # calculate the heat conductivity and the heat capacity
        property_h, property_c = mixture_property(Th, pi_h), mixture_property(Tc, pi_c)
        mix_pro_ave = (property_h + property_c) / 2
        # print(mix_pro_ave)
        k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator
        cal_property = [mix_pro_ave["cp_Methanol"], mix_pro_ave["cp_H2O"], 4.5e-5, 1.4e-5, k_e]

        # diffusional governing equation
        res_guess = diff.ode_multi(cond_list[position], cond_list[position - 1], P, cal_property, r_CH3OH_H20)

        gap_xd = res_guess[1][-1] - xi_h["H2O"]
        if abs(gap_xd) < gap_min:
            gap_min = abs(gap_xd)
            r_sel = r_CH3OH_H20
            if gap_min / xi_h["H2O"] < 0.05: break
        # if gap_xd > 0:
        #     rmax = r_sel
        # else:
        #     rmin = r_sel
    # calculate the diffusional flux with optimized N_CH3OH/N_H2O
    pi_c_cond = diff.p_sat(Tc, [r_sel / (1 + r_sel), 1 / (1 + r_sel)])
    pi_c = diff.cold_comp(pi_h, pi_c_cond)
    xi_c = pi_c / P
    cold_cond = [Tc, xi_c["Methanol"], xi_c["H2O"]]
    hot_cond = [Th, xi_h["Methanol"], xi_h["H2O"]]
    cond_list = [hot_cond, cold_cond]
    hot_cond.append(radium[position]), cold_cond.append(radium[1 - position])
    res = diff.ode_multi(cond_list[position], cond_list[position - 1], P, cal_property, r_sel)
    # [xc, xd, Nd, T, dTdz]
    # xsol = np.linspace(0.03 / 2, 0.07 / 2, 200)
    # fig, axe = plt.subplots(2, 2)
    # axe[0][0].plot(xsol, res[0])
    # axe[0][0].plot(xsol, res[1])
    # axe[0][0].legend(["CH3OH", "H2O"])
    # axe[0][1].plot(xsol, res[2])
    # axe[1][0].plot(xsol, res[3])
    # axe[1][1].plot(xsol, res[4])
    # print(res)
    # plt.show()

    na_H20 = res[2][-position] * radium[-position] * 2 * np.pi
    na_CH3OH = na_H20 * r_sel
    qcv = -k_e * res[4][-position] * radium[-position] * 2 * np.pi

    dT = qcv / Ft0 / property_h["cp_m"]  # k/m
    dev = gap_min / xi_h["H2O"]
    return na_H20, na_CH3OH, dT, dev


ks = 0.2  # W/m K
vof = 0.8
R = 8.314

# a = [150.3101838,489.1689634,19.85134442,46.12569765,19.11920599]
# # a = [221.0032744,718.0968433,29.92141444,63.73702749,27.54351001]
# # a = [298.0825362,907.4924509,17.0857909,23.59907537,6.622421091]
# # a = [308.1391103, 931.9010154, 10.47495045, 14.13205543, 3.74184218]
# # a = [0.209023659, 0.661347507, 0.060604809, 0.051885759, 0.017138265]
# F = pd.Series(a, index=["CO2", "H2", "Methanol", "H2O", "CO"])
# f1, f2, f3 = open('in_chem.json'), open('in_reactor.json'), open('in_feed.json')
# chem_dict = json.load(f1)
# reactor_dict = json.load(f2)
# feed_dict = json.load(f3)
# print(flux(518, 50, F, reactor_dict["insulator"]))
#
