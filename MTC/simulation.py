import numpy as np
import json
import matplotlib.pyplot as plt

import pandas as pd
import balance as ba
import insulator as ins

R = 8.314


def simulator(chem_data, reactor_data, feed_data):
    # reactor parameters
    L, Dt = reactor_data["reactor"]['L'], reactor_data["reactor"]['Dt']  # length, m
    nrt = reactor_data["reactor"]['nt']  # number of the reaction tube
    phi = reactor_data["reactor"]["phi"]  # void of fraction
    rhoc = reactor_data["reactor"]["rhoc"]  # density of catalyst, kg/m3
    insulator_para = reactor_data["insulator"]  # reaction occurring in the inside or outside of insulator
    nit = insulator_para["nt"]  # tube number of the insulator

    # parameters of feed gas
    F0 = {}
    P0, T0 = feed_data["condition"]["P"], feed_data["condition"]["T"]  # P0 bar, T0 K
    v0 = feed_data["condition"]["Sv"] / nrt  # volumetric flux per tube, m3/s
    # print(feed_data["condition"]["Sv"])

    Ft0 = P0 * 1e5 * v0 / R / T0  # total flux of feed,mol/s
    if feed_data["condition"]["recycle"] == "off":  # recycled stream
        F0["CO2"] = Ft0 / (feed_data["condition"]["H2/CO2"] + 1)
        F0["H2"] = Ft0 - F0["CO2"]
        for comp in chem_data["comp_list"]: F0[comp] = 0 if comp != "CO2" and comp != "H2" else F0[comp]
    elif feed_data["condition"]["recycle"] == "on":  # fresh stream
        n = 0
        feed = [float(i) for i in feed_data["feed"].split('\t')]
        for comp in chem_data["comp_list"]:
            F0[comp] = feed[n]
            n += 1

    feed_info = {"T": T0, "P": P0, "Sv": v0, "Ft0": Ft0}
    print(feed_info)
    print(chem_data)

    # create the mesh along the tube
    N_s, N_m, N_l, z_s = 100001, 10001, 10001, 1e-6
    N = N_s+N_l #+N_m
    # grid_s = np.logspace(np.log10(1e-15), np.log10(z_s), N_s)
    grid_s = np.linspace(0, z_s, N_s)
    grid_m = np.linspace(0, z_s, N_s)
    grid_l = np.logspace(np.log10(z_s), np.log10(L), N_l)
    grid = np.hstack((grid_s, grid_l[1:]))
    dl = grid[1:] - grid[:-1]

    col_list = ["z", "T", "P", "qcv"] + ['F' + temp for temp in chem_data["comp_list"]]  # "q", "cp","dT_diff","dF_diff"
    sim_data = pd.DataFrame(index=np.arange(N-1), columns=col_list)  # store simulated result
    sim_data['z'] = grid

    # prepare data for 4th order Runge Kutta
    P = P0
    T = np.array([T0, T0, T0, T0]).astype("float")

    dT_np = np.zeros(4)
    dF_pd = pd.DataFrame(index=[1, 2, 3, 4], columns=chem_data["comp_list"])
    F_pd = pd.DataFrame([list(F0.values())] * 4, index=[1, 2, 3, 4], columns=chem_data["comp_list"])
    # print(F_pd.iloc[0])
    scale = [0.5, 0.5, 1, 1]

    # perform 4th order Runge Kutta calculation along the tube
    for i in range(N_s):
        print(i, dl[i], sim_data['z'].iloc[i])
        dl2dw = (np.pi * Dt ** 2 / 4) * dl[i] * (1 - phi) * rhoc  # convert per length to per kg catalyst
        # if on, calculate the diffusional flux across the insulator
        # if i % 100 == 0: print(i)
        if insulator_para["status"] == "on":
            dF_diff, dT_diff, qcv = ins.mflux(T[0], P, F_pd.iloc[0], insulator_para)
            F_pd.loc[1, "H2O"] -= abs(dF_diff * dl * nit)
            T[0] -= abs(dT_diff * dl * nit)

        sim_data.iloc[i, 1:4] = [T[0], P0, 0]
        sim_data.iloc[i, 4:] = F_pd.iloc[0]

        F_pd.iloc[1] = F_pd.iloc[2] = F_pd.iloc[3] = F_pd.iloc[0]
        T[1] = T[2] = T[3] = T[0]
        for j in range(4):

            # calculate the temperature variation and the change of the molar flow rate due to reactions, mol/s
            dT_np[j], dF_pd.iloc[j] = ba.balance(T[j], P, F_pd.iloc[j], feed_info, chem_data)  ## * dl2dw
            dT_np[j] *= dl2dw
            dF_pd.iloc[j] *= dl2dw

            if j == 3:
                for comp in chem_data["comp_list"]:
                    F_pd.loc[1, comp] += (dF_pd.loc[1, comp] + 2 * dF_pd.loc[2, comp]
                                          + 2 * dF_pd.loc[3, comp] + dF_pd.loc[4, comp]) / 6
                T[0] += np.sum(dT_np * np.array([1, 2, 2, 1])) / 6
            else:
                F_pd.iloc[j + 1] += dF_pd.iloc[j] * scale[j]
                T[j + 1] = T[j] + dT_np[j] * scale[j]

    sim_data["S_MeOH/CO"] = sim_data["FMethanol"] / (sim_data["FCO"] + 1e-10)
    sim_data["Conversion_CO2"] = (1 - sim_data["FCO2"] / F0["CO2"])
    return sim_data


f1, f2, f3 = open('in_chem.json'), open('in_reactor.json'), open('in_feed.json')
chem_dict = json.load(f1)
reactor_dict = json.load(f2)
feed_dict = json.load(f3)
path = "result/result_mod_mesh_keq_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.xlsx" % \
       (feed_dict["condition"]["recycle"], feed_dict["condition"]["Sv"],
        feed_dict["condition"]["T"], feed_dict["condition"]["P"],
        reactor_dict["reactor"]["Dt"], reactor_dict["reactor"]["nt"],
        reactor_dict["insulator"]["status"], reactor_dict["insulator"]["Tc"],
        reactor_dict["insulator"]["Din"], reactor_dict["insulator"]["nt"])
result = simulator(chem_dict, reactor_dict, feed_dict)
# print(path)
result.to_excel(path)
# print(result)
fig, axes = plt.subplots(5, 1, sharex=True)
axes[0].plot(result["z"], result["FCO2"])
# axes[0].set_title("F_CO2")
# plt.ylabel("mol/m3")
# plt.show()
axes[1].plot(result["z"], result["FH2O"])
# axes[1].set_title("F_H2O")
# plt.show()
axes[2].plot(result["z"], result["FMethanol"])
# axes[2].set_title("F_Methanol")
# plt.show()
axes[3].plot(result["z"], result["Conversion_CO2"])
# axes[3].set_title("Conversion_CO2")
# plt.show()

axes[4].plot(result["z"], result["T"])
# axes[4].set_title("T")
# plt.tight_layout()
plt.show()
