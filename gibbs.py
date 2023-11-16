import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from prop_calculator import VLE, Thermo
from CoolProp.CoolProp import PropsSI


class Gibbs:
    def __init__(self, comp, elements=None):
        self.comp = comp
        # elements counts for each comps
        self.elements = self.__elements() if elements is None else elements
        self.g_cal = Thermo()

    @staticmethod
    def __elements():
        return np.array([[1, 0, 1, 0, 1],
                         [0, 2, 4, 2, 0],
                         [2, 0, 1, 1, 1]])

    @staticmethod
    def cal_fi(T, P, x, eos=1):
        """
        calculate fi for mix at T P
        :param T: K
        :param P: bar
        :param x: molar flow rate (pd.Series), mol/s
        :param eos:
        :return: partial fugacity (pd.Series), bar; compression factor
        """
        if (x < 0).any():
            return x
        mol_frac = x / x.sum()

        # pure fluid
        if len(x) == 1:
            Tc = PropsSI('Tcrit', x.index) * 1e-5  # K
            Psat = PropsSI('P', 'T', T, 'Q', 1, x.index) * 1e-5 if T < Tc else 1e5  # bar
            fi_cal = VLE(T, x.index.tolist())
            if P >= Psat:
                # liquid
                phi, _ = fi_cal.phi(mol_frac, P, phase=1)
            else:
                # vapor
                phi, _ = fi_cal.phi(mol_frac, P, phase=0)
            fi = mol_frac * P * phi
            return fi

        # mixture
        if eos == 1:
            # generate VLE calculator
            fi_cal = VLE(T, x.index.tolist())

            # evaluate pressure of dew point
            if "Methanol" in x.index.tolist() and "H2O" in x.index.tolist() and\
                    mol_frac["Methanol"] + mol_frac["H2O"] > 1e-5:
                # using the mixture of CH3OH+H2O to guess the dew point of the whole mix
                mol_guess = mol_frac[["Methanol", "H2O"]] / mol_frac[["Methanol", "H2O"]].sum()
                P_dew_cal = VLE(T, ["Methanol", "H2O"])
                P_dew_cond = P_dew_cal.dew_p(mol_guess)['P']
                P_dew = P_dew_cond / mol_frac[["Methanol", "H2O"]].sum() * mol_frac.sum()
            else:
                P_dew = 1000

            if P < P_dew:
                # vapor phase
                phi, _ = fi_cal.phi(mol_frac, P, phase=0)
                fi = mol_frac * P * phi
            else:
                if mol_frac["Methanol"] + mol_frac["H2O"] < 1 - 1e-5:
                    # VLE
                    comp, phi = fi_cal.flash(P, mol_frac)  # liquid phase phi
                    fi = comp.loc['L1'] / comp.loc['L1'].sum() * P * phi
                else:
                    # liquid phase, there are only CH3OH+H2O
                    fi = pd.Series(0, index=x.index)
                    mol_frac_l = mol_frac[["Methanol", "H2O"]] / (mol_frac["Methanol"] + mol_frac["H2O"])
                    phi, _ = P_dew_cal.phi(mol_frac_l, P, phase=1)
                    fi.loc[["Methanol", "H2O"]] = mol_frac_l * P * phi
        elif eos == 0:
            fi = mol_frac * P
        return fi

    @staticmethod
    def eq_cons(x, feed, element_count):
        """
        conservation of matter
        """
        return [np.sum((x - feed) * element_count[i]) for i in range(3)]

    def mix_gibbs(self, T, P, x, eos=1):
        """
        calculate the gibbs energy of mixture at give T P
        :param T: K
        :param P: bar
        :param x: molar flow of comps in mixture, mol/s (np.array)
        :param eos:
        :return: total gibbs energy of mixture, kW
        """
        if (x < 0).any():
            return 0
        else:
            fi = self.cal_fi(T, P, pd.Series(x, index=self.comp), eos=eos)
            gi = self.g_cal.G(T, fi)
            return np.dot(x, gi)

    def min_eq(self, T, P, feed, x0=None):
        """
        calculate the product by minimizing the Gibbs energy
        :param T: K
        :param P: bar
        :param feed: molar flow rate (pd.Series), mol/s
        :param x0: initial guess of product
        :return:
        """
        # Define the initial guess for the molar fractions of the product
        if x0 is None:
            x0 = feed.copy()
            r_guess, s_guess = 0.5, 0.5
            x0.loc[:] = np.array([(1 - r_guess) * x0[0],
                                  x0[1] - x0[0] * (2 * r_guess * s_guess + r_guess),
                                  r_guess * s_guess * x0[0],
                                  r_guess * x0[0],
                                  r_guess * (1 - s_guess) * x0[0]])
        else:
            x0 = x0
        # Combine equality and inequality constraints
        constraint_rule = [{'type': 'eq', 'fun': lambda x: self.eq_cons(x, feed, self.elements)},
                           {'type': 'ineq', 'fun': lambda x: x}]  # ensure >= 0

        # Solve the optimization problem using SciPy's minimize function
        res = minimize(lambda x: self.mix_gibbs(T, P, x, eos=1), x0, constraints=constraint_rule,
                       method="SLSQP")
        return res.x, res.fun

    def solve_over_range(self, T, P, feed_comp, save=False):
        """
        calculate the equilibrium conversion over ranges of temperature and pressure
        """
        CO2_R = pd.DataFrame(index=T, columns=P)
        select = pd.DataFrame(index=T, columns=P)
        for P_eq in P:
            for T_eq in T:
                print(T_eq, P_eq)
                # Calculate equilibrium
                product, min_gibbs_energy = self.min_eq(T_eq, P_eq, feed_comp)
                CO2_R.loc[T_eq, P_eq] = (feed_comp[0] - product[0]) / feed_comp[0]
                print(CO2_R.loc[T_eq, P_eq])
                select.loc[T_eq, P_eq] = (product[2] - feed_comp[2]) / (feed_comp[0] - product[0])
        if save:
            res_path = 'res_Gibbs/eq_%s_%s_%s_%s.xlsx' % (min(T_eqs), max(T_eqs), min(P_eqs), max(P_eqs))
            with pd.ExcelWriter(res_path, engine='openpyxl') as writer:
                CO2_R.to_excel(writer, index=True, header=True, sheet_name='conversion')
                select.to_excel(writer, index=True, header=True, sheet_name='select')
        return CO2_R, select

    def series_reactor(self, feed_comp, T, P, r_target):
        """
        the combination of Gibbs reactor and separator
        :param feed_comp: molar flow rate (pd.Series), mol/s
        :param T: K
        :param P: bar
        :param r_target: the target of conversion of CO2
        :return: product in each stage
        """
        r_t = 0
        products, r_each = [], []
        reactor_feed = feed_comp
        sep_work, dHrs = [], []
        while r_t < r_target:
            product, _ = self.min_eq(T, P, feed_comp)

            # metric of reactor
            dHr = np.dot(self.g_cal.H(T, self.comp), (product - reactor_feed))  # dH during reaction, kW
            r = (reactor_feed.loc['CO2'] - product.loc['CO2']) / reactor_feed.loc['CO2']  # CO2 conversion
            r_each.append([r.values])
            r_t = (feed_comp.loc['CO2'] - product.loc['CO2']) / feed_comp.loc['CO2']  # total conversion of CO2

            products.append(product.tolist())
            reactor_feed = product.copy()
            reactor_feed[2], reactor_feed[3] = 0, 0

            # metric of separator
        #     Wf =
        #
        #     sep_work.append([Ws, Wc, We])
        #     dHrs.append([dHr])
        #
        # sep_work = np.array(sep_work)
        # sep_work = np.vstack((sep_work, np.sum(sep_work, axis=0)))
        # dHrs = np.array(dHrs) * (1 - 298.15 / temperature)  # exergy
        # dHrs = np.vstack((dHrs, np.sum(dHrs, axis=0)))
        # r_each = np.array(r_each)
        # r_each = np.vstack((r_each, r))
        # res_sim = np.hstack((r_each, sep_work, dHrs))
        # res_sim = pd.DataFrame(res_sim, index=np.arange(len(dHrs)) + 1, columns=['r', 'Ws', 'Wc', 'We', 'Qr'])

        # return res_sim

    def metric(self, F1, T, P):
        """
        calculate the metric of separator
        F1 is split into F2 and F3
        :param F1: molar flow rate (pd.Series), mol/s
        :param T: K
        :param P: bar
        :return:
        """
        F2 = F1.loc[["CO2", "H2", "CO"]]
        F3 = F1.loc[["Methanol", "H2O"]]
        F = [F1, F2, F3]
        G, H = np.zeros(3), np.zeros(3)

        for i in range(3):
            fi = self.cal_fi(T, P, F[i], eos=1)
            G[i] = np.dot(self.g_cal.G(T, fi), F[i])
            H[i] = np.dot(self.g_cal.H(T, fi.index), F[i])
        print(G)
        print(G[1] + G[2] - G[0])
        print(H)
        print(H[1]+H[2]-H[0])


in_gas = pd.Series([1, 3, 0, 0, 0], index=["CO2", "H2", "Methanol", "H2O", "CO"])
# calculate the equilibrium performance over wide range
T_eqs = np.arange(438, 588, 10)
P_eqs = np.array([70])  # np.arange(30, 110, 20)

gibbs_cal = Gibbs(in_gas.index)
print(gibbs_cal.min_eq(473, 70, in_gas))
# gibbs_cal.solve_over_range(T_eqs, P_eqs, in_gas,save=True)
# sep_in = pd.Series([0.665144199678156, 2.06715255801167,
#                     0.298995820833231, 0.334855800321855, 0.0358599794534844],
#                    index=["CO2", "H2", "Methanol", "H2O", "CO"])
# gibbs_cal.metric(sep_in, 503, 70)
