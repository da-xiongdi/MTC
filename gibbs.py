import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
# import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from prop_calculator import VLE


class EquilibriumCalculator:
    def __init__(self, hf_params, gf_params, temperature, pressure, eos=0):
        self.hf_params = hf_params
        self.gf_params = gf_params
        self.temperature = temperature
        self.pressure = pressure
        self.eos = eos  # 0 for ideal 1 for SRK

        # Calculate enthalpy of formation, and Gibbs energy of formation
        # at standard state
        self.dHfs_0 = np.array([self.cal_df(hf_params.loc[substance], 298.15) for substance in hf_params.index])
        self.dGfs_0 = np.array([self.cal_df(gf_params.loc[substance], 298.15) for substance in hf_params.index])
        # at temperature
        self.dHfs_T = np.array([self.cal_df(hf_params.loc[substance], temperature) for substance in hf_params.index])
        self.dGfs_T = np.array([self.cal_df(gf_params.loc[substance], temperature) for substance in hf_params.index])

    @staticmethod
    def cal_df(f_data, T):
        """
        calculate the energy (Gibbs energy or enthalpy) of formation for a species at T
        """
        fit_para = np.polyfit(f_data.index, f_data.values, 2)
        df = np.polyval(fit_para, T)
        return df

    def cal_dr(self, f_data, feed, product, T):
        """
        calculate the energy (Gibbs energy or enthalpy) change of a reaction at T
        """
        fs = np.zeros(len(f_data.index))
        i = 0
        for substance in f_data.index:
            fs[i] = self.cal_df(f_data.loc[substance], T)
            i += 1
        return np.sum(fs * (product - feed))

    @staticmethod
    def cal_gibbs(x, temperature, pressure, gf_paras, eos=1, p=1):
        """
        calculate the gibbs energy of the pure fluid
        :param x: flow rate of comps (pd.Serize), mol/s
        :return: gibbs energy, kW
        """
        for n in x:
            if n < 0:
                return 0
        if eos == 1:
            # generate VLE calculator
            fi_cal = VLE(temperature, x.index.tolist())
            phi, _ = fi_cal.phi(x, pressure, phase=p)
            fi = pressure * phi
        elif eos == 0:
            fi = pressure
        fit_para = np.polyfit(gf_paras.index, gf_paras.values, 2)
        dGfs_T = np.polyval(fit_para, temperature)
        dGrs = (dGfs_T + 8.314 * temperature * np.log(fi / 1) / 1000) * x.values
        print(dGrs)
        print(fi)
        return dGrs

    def mix_gibbs(self, x):
        """
        calculate the gibbs energy of the mixture
        :param x: flow rate of comps (ndarray), mol/s
        :return: gibbs energy, kW
        """
        for n in x:
            if n < 0:
                return 0
        mol_frac = pd.Series(x / np.sum(x), index=self.hf_params.index)
        if self.eos == 1:
            # generate VLE calculator
            fi_cal = VLE(self.temperature, self.hf_params.index.tolist())
            # evaluate pressure of dew point
            # guess the vapor phase
            mol_guess = mol_frac[["Methanol", "H2O"]] / np.sum(mol_frac.values[2:4])
            P_dew_cal = VLE(self.temperature, ["Methanol", "H2O"])
            # print(mol_guess)
            P_dew_cond = P_dew_cal.dew_p(mol_guess)['P']
            P_dew = P_dew_cond / np.sum(mol_frac.values[2:4]) * np.sum(mol_frac.values)
            # print(P_dew)
            # print(mol_frac)
            if self.pressure < P_dew:
                # vapor phase
                phi, _ = fi_cal.phi(mol_frac, self.pressure)
                fi = mol_frac * self.pressure
            else:
                if mol_frac[2] + mol_frac[3] < 1 - 1e-5:
                    # VLE
                    # print('VLE')
                    comp, phi = fi_cal.flash(self.pressure, mol_frac)  # liquid phase phi
                    fi = comp.loc['L1'].values / np.sum(comp.loc['L1'].values) * self.pressure * phi
                else:
                    # liquid phase
                    # print('L')
                    mol_frac_l = mol_frac[2:4] / (mol_frac[2] + mol_frac[3])
                    phi, _ = P_dew_cal.phi(mol_frac_l, self.pressure, phase=1)
                    fi_l = mol_frac_l * self.pressure * phi
                    fi = np.array([0, 0] + fi_l.tolist() + [0])
        elif self.eos == 0:
            fi = mol_frac * self.pressure
        dGrs = 0
        for j in range(len(x)):
            if fi[j] <= 1e-15:
                dGrs += 0
            else:
                dGrs += (self.dGfs_T[j] + 8.314 * self.temperature * np.log(fi[j] / 1) / 1000) * x[j]
        return dGrs

    @staticmethod
    def equality_constraints(x, feed, element_count):
        return [np.sum((x - feed) * element_count[i]) for i in range(3)]

    def minimize_equilibrium(self, feed, element_count, x0=None):
        # Define the initial guess for the molar fractions of the product
        if x0 is None:
            x0 = feed.copy()
            r_guess, s_guess = 0.5, 0.5
            x0[2:] = x0[0] * np.array([r_guess * s_guess,
                                       r_guess,
                                       r_guess * (1 - s_guess)])
            x0[:2] = np.array([(1 - r_guess) * x0[0],
                               x0[1] - x0[0] * (2 * r_guess * s_guess + r_guess)])
            # x0 = feed.copy()
            # x0[2:] = x0[0] * np.array([0.2, 0.3, 0.1])
            # x0 = x0 * np.array([0.7, 23 / 30, 1, 1, 1])
        else:
            x0 = x0

        # Define bounds for the variables (product fractions should be between 0 and 1)
        # bounds = [(0, 1) for _ in range(len(x0))]
        # Combine equality and inequality constraints
        constraint_rule = [{'type': 'eq', 'fun': lambda x: self.equality_constraints(x, feed, element_count)},
                           {'type': 'ineq', 'fun': lambda x: x}]  # ensure >= 0

        # Solve the optimization problem using SciPy's minimize function
        result = minimize(lambda x: self.mix_gibbs(x), x0, constraints=constraint_rule, method="SLSQP")
        return result.x, result.fun


def solve_over_range(temperatures, pressures):
    """
    calculate the equilibrium conversion over ranges of temperature and pressure
    """
    CO2_R = pd.DataFrame(index=temperatures, columns=pressures)
    select = pd.DataFrame(index=temperatures, columns=pressures)
    for P_eq in pressures:
        p_guess = None
        for T_eq in temperatures:
            print(T_eq, P_eq)
            eq_calculator = EquilibriumCalculator(Hfs, Gfs, T_eq, P_eq, 1)
            # Calculate equilibrium
            product, min_gibbs_energy = eq_calculator.minimize_equilibrium(feed_comp, element, p_guess)

            CO2_R.loc[T_eq, P_eq] = (feed_comp[0] - product[0]) / feed_comp[0]
            print(CO2_R.loc[T_eq, P_eq])
            select.loc[T_eq, P_eq] = (product[2] - feed_comp[2]) / (feed_comp[0] - product[0])
            # p_guess = feed_comp.copy()
            # r_guess, s_guess = round(CO2_R.loc[T_eq, P_eq] * 0.8, 4), round(select.loc[T_eq, P_eq] * 0.8, 4)
            # p_guess[2:] = p_guess[0] * np.array([r_guess * s_guess,
            #                                      r_guess,
            #                                      r_guess * (1 - s_guess)])
            # p_guess[:2] = np.array([(1 - r_guess) * p_guess[0],
            #                         p_guess[1] - p_guess[0] * (2 * r_guess * s_guess + r_guess)])
            # print(eq_calculator.equality_constraints(p_guess, feed_comp, element))

    # plt.ylim(0, 1)
    # for k in range(len(pressures)):
    #     plt.plot(temperatures - 273.15, CO2_R[k], label=pressures[k])
    #     # plt.plot(temperatures - 273.15, select[0], label=pressures[0])
    # plt.legend()
    # plt.show()
    return CO2_R, select


def series_reactor(feed, temperature, pressure, r_target):
    """
    the combination of Gibbs reactor and separator
    :param feed: molar flow rate of comps
    :param temperature: K
    :param pressure: bar
    :param r_target: the target of conversion of CO2
    :return: product in each stage
    """
    r = 0
    products, r_reactor = [], []
    reactor_feed = feed
    sep_work, dHrs = [], []
    while r < r_target:
        eq_calculator = EquilibriumCalculator(Hfs, Gfs, temperature, pressure, 1)
        product, _ = eq_calculator.minimize_equilibrium(reactor_feed, element)
        dHr = -eq_calculator.cal_dr(Hfs, reactor_feed, product, temperature)  # heat release during reaction, kW
        r_reactor.append([(reactor_feed[0] - product[0]) / reactor_feed[0]])  # CO2 conversion of each reactor
        r = (feed[0] - product[0]) / feed[0]  # total conversion of CO2
        products.append(product.tolist())
        reactor_feed = product.copy()
        reactor_feed[2], reactor_feed[3] = 0, 0

        FR = [product.copy(), temperature, pressure]
        F1 = [reactor_feed, temperature, np.sum(reactor_feed) / np.sum(product) * pressure]
        F2_comp = np.array([0, 0, product[2], product[3], 0])
        F2 = [F2_comp, temperature, np.sum(F2_comp) / np.sum(product) * pressure]
        F3 = [reactor_feed, temperature, pressure]
        F4 = [F2_comp, 298.15, 1.0]

        Ws = -Work_min([FR, F1, F2])  # minus means work output
        Wc = -Work_min([F1, F3])
        We = -Work_min([F2, F4])
        sep_work.append([Ws, Wc, We])
        dHrs.append([dHr])

    sep_work = np.array(sep_work)
    sep_work = np.vstack((sep_work, np.sum(sep_work, axis=0)))
    dHrs = np.array(dHrs) * (1 - 298.15 / temperature)  # exergy
    dHrs = np.vstack((dHrs, np.sum(dHrs, axis=0)))
    r_reactor = np.array(r_reactor)
    r_reactor = np.vstack((r_reactor, r))
    res_sim = np.hstack((r_reactor, sep_work, dHrs))
    res_sim = pd.DataFrame(res_sim, index=np.arange(len(dHrs)) + 1, columns=['r', 'Ws', 'Wc', 'We', 'Qr'])

    return res_sim


def Work_min(F_info):
    """
    work released or required from stat 1 to stat 2
    :param F_info: gases info including comps, temperature, and pressure [info1, info2, ...]
    """
    gibbs_energy = np.zeros(len(F_info))
    for i in range(len(F_info)):
        gibbs_cal = EquilibriumCalculator(Hfs, Gfs, F_info[i][1], F_info[i][2], 1)
        gibbs_energy[i] = gibbs_cal.mix_gibbs(F_info[i][0])
    return gibbs_energy[0] - np.sum(gibbs_energy[1:])


def mix_gibbs(feed, temperature, pressure):
    ideal_cal = EquilibriumCalculator(Hfs, Gfs, temperature, pressure, 0)
    ideal_gibbs = ideal_cal.mix_gibbs(feed)
    real_cal = EquilibriumCalculator(Hfs, Gfs, temperature, pressure, 1)
    real_gibbs = real_cal.mix_gibbs(feed)
    print(ideal_gibbs, real_gibbs, ideal_gibbs - real_gibbs)


if __name__ == "__main__":
    # # Parameters for calculating enthalpy of formation
    # # CO2 H2 CH3OH H2O CO
    Ts = np.arange(300, 1100, 100)
    Gfs = np.array([[-394.379, -394.656, -394.914, -395.152, -395.367, -395.558, -395.724, -395.865],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [-162.057, -148.509, -134.109, -119.125, -103.737, -88.063, -72.188, -56.170],
                    [-228.5, -223.9, -219.05, -214.008, -208.814, -203.501, -198.091, -192.603],
                    [-137.333, -146.341, -155.412, -164.480, -173.513, -182.494, -191.417, -200.281]])

    Hfs = np.array([[-393.511, -393.586, -393.672, -393.791, -393.946, -394.133, -394.343, -394.568],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [-201.068, -204.622, -207.750, -210.387, -212.570, -214.350, -215.782, -216.916],
                    [-241.844, -242.845, -243.822, -244.751, -245.620, -246.424, -247.158, -247.820],
                    [-110.519, -110.121, -110.027, -110.157, -110.453, -110.870, -111.378, -111.952]])
    Gfs = pd.DataFrame(Gfs, index=["CO2", "H2", "Methanol", "H2O", "CO"], columns=Ts)
    Hfs = pd.DataFrame(Hfs, index=["CO2", "H2", "Methanol", "H2O", "CO"], columns=Ts)

    # # Define the feed and product compositions
    feed_comp = np.array([0.00825, 0.02475, 0, 0, 0])  # CO2 H2 CH3OH H2O CO
    product_comp = np.array([0, 0, 1, 1, 0])  # CO2 H2 CH3OH H2O CO

    element = np.array([[1, 0, 1, 0, 1],
                        [0, 2, 4, 2, 0],
                        [2, 0, 1, 1, 1]])

    # calculate the equilibrium performance over wide range
    # T_eqs = np.arange(438, 588, 10)
    # P_eqs = np.array([50])  # np.arange(30, 110, 20)
    # r_eq, s_eq = solve_over_range(T_eqs, P_eqs)
    # rec_path = r"D:\document\00Study\05多联产系统\甲醇单元\Gibbs\sim\eq_%s_%s_%s_%s.xlsx" % \
    #            (min(T_eqs), max(T_eqs), min(P_eqs), max(P_eqs))
    # with pd.ExcelWriter(rec_path, engine='openpyxl') as writer:
    #     r_eq.to_excel(writer, index=True, header=True, sheet_name='conversion')
    #     s_eq.to_excel(writer, index=True, header=True, sheet_name='select')

    # test1 = EquilibriumCalculator(Hfs, Gfs, 523, 8, eos=1)
    # g1 = test1.mix_gibbs(np.array([0, 0, 0.5, 0.5, 0]))
    # f = pd.Series(1, index=["H2O"])
    # g1 = EquilibriumCalculator.cal_gibbs(f, 454, 10, Gfs.loc['H2O'], eos=1, p=0)
    # g2 = EquilibriumCalculator.cal_gibbs(f, 454, 15, Gfs.loc['H2O'], eos=1, p=1)
    # print(g1 - g2)
    #
    # g2 = EquilibriumCalculator.cal_gibbs(pd.Series(0.5, index=["Methanol"]), 300, 1, Gfs.loc['Methanol'])
    # g3 = EquilibriumCalculator.cal_gibbs(pd.Series(0.5, index=["H2O"]), 300, 1, Gfs.loc['H2O'])
    # print(g1, g2, g3)

    sim_T, sim_P, sim_r = 503, 70, 0.3
    res = series_reactor(feed_comp, sim_T, sim_P, r_target=sim_r)
    res_path = r"D:\document\00Study\05多联产系统\甲醇单元\Gibbs\sim\res_%s_%s_%s.xlsx" % (sim_T, sim_P, sim_r)
    res.to_excel(res_path, index=True, header=True, engine='openpyxl')

    # ss = EquilibriumCalculator(Hfs, Gfs, 423, 80, eos=1)
    # gases = np.array([1,3,0.5,0.5,1])
    # print(ss.mix_gibbs(gases))
    # plt.plot(stages, work[:, 0], label="Ws")
    # plt.plot(stages, work[:, 1], label="Wc")
    # plt.plot(stages, qr, label="Qr")
    # plt.legend()
    # plt.show()
    # Ws_t = np.sum(work[:, 0])
    # Wc_t = np.sum(work[:, 1])
    # print(Ws_t, Wc_t, Ws_t + Wc_t, np.sum(qr)*(1-298.15/493))

    # p1 = np.array([0.751991, 2.31926, 0.216366, 0.248009, 0.0316429]) # data from Aspen plus
