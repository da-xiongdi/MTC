import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from prop_calculator import VLEThermo

from insulator import Insulation
from thermo import (ChemicalConstantsPackage, SRKMIX, FlashVL, CEOSLiquid, CEOSGas, HeatCapacityGas,
                    GibbsExcessLiquid, MSRKMIX, SRKMIXTranslated, SRK)


class Gibbs(VLEThermo):
    def __init__(self, comp, ref='ch', elements=None):
        super().__init__(comp, ref)
        self.comp = comp
        # elements counts for each comps
        self.elements = self.__elements() if elements is None else elements
        self.const, self.cor = ChemicalConstantsPackage.from_IDs(self.comp.tolist())
        self.cp, self.eos_kw = self.eos_paras()

    @staticmethod
    def __elements():
        return np.array([[1, 0, 1, 0, 1],
                         [0, 2, 4, 2, 0],
                         [2, 0, 1, 1, 1]])

    def cal_Gr(self, T, P, sto):
        sto = np.array(sto)
        feed = np.where(sto <= 0, sto, 0)
        product = np.where(sto > 0, sto, 0)
        return self.cal_G(T, P, product) + self.cal_G(T, P, feed)

    def cal_Hr(self, T, P, sto):
        sto = np.array(sto)
        feed = np.where(sto <= 0, sto, 0)
        product = np.where(sto > 0, sto, 0)
        return self.cal_H(T, P, product) + self.cal_H(T, P, feed)

    def cal_Sr(self, T, P, sto):
        sto = np.array(sto)
        feed = np.where(sto <= 0, sto, 0)
        product = np.where(sto > 0, sto, 0)
        return self.cal_S(T, P, product) + self.cal_S(T, P, feed)

    def cal_dH(self, T, P, x):
        frac = x / np.sum(x)
        gas = CEOSGas(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw, T=T, P=P, zs=frac)
        H_dep_in = gas.H_dep()
        liquid = CEOSLiquid(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw)
        flasher = FlashVL(self.const, self.cor, liquid=liquid, gas=gas)
        PT = flasher.flash(zs=frac, T=T, P=P * 1E5)
        H_dep = PT.H()
        return (H_dep_in - H_dep) * np.sum(x) / 1000  # kW

    def cal_dn(self, T, P, x):
        frac = x / np.sum(x)
        gas = CEOSGas(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw, T=T, P=P, zs=frac)
        liquid = CEOSLiquid(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw)
        flasher = FlashVL(self.const, self.cor, liquid=liquid, gas=gas)
        PT = flasher.flash(zs=frac, T=T, P=P * 1E5)
        # gas_sep = PT.gas.zs
        liq_sep = PT.liquid0.zs

        return (1 - PT.VF) * np.sum(x) * np.array(liq_sep)  # mol/s

    @staticmethod
    def eq_cons(x, feed, element_count):
        """
        conservation of matter
        """
        return [np.sum((x - feed) * element_count[i]) for i in range(3)]

    def mix_gibbs(self, T, P, x):
        """
        calculate the gibbs energy of mixture at give T P
        :param T: K
        :param P: bar
        :param x: molar flow of comps in mixture, mol/s (np.array)
        :return: total gibbs energy of mixture, kW
        """
        x = np.array(x)
        if (x < 0).any():
            return 0
        else:
            return self.cal_G(T, P, x)

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
        res = minimize(lambda x: self.mix_gibbs(T, P, x), x0, constraints=constraint_rule,
                       method="SLSQP")
        return res.x, res.fun

    def solve_over_range(self, T, P, feed_comp, save=False):
        """
        calculate the equilibrium conversion over ranges of temperature and pressure
        :param T: K
        :param P: bar
        :param feed_comp: molar flow rate (pd.Series), mol/s
        :param save: whether save data to file
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
            res_path = 'res_Gibbs/eq_MSRK_%s_%s_%s_%s.xlsx' % (min(T), max(T), min(P), max(P))
            with pd.ExcelWriter(res_path, engine='openpyxl') as writer:
                CO2_R.to_excel(writer, index=True, header=True, sheet_name='conversion')
                select.to_excel(writer, index=True, header=True, sheet_name='select')
        return CO2_R, select

    def series_reactor(self, feed_comp, T, P, r_target, sp_paras, sf_target=0.95):
        """
        the combination of Gibbs reactor and separator
        :param feed_comp: molar flow rate (pd.Series), mol/s
        :param T: feed temperature, K
        :param P: bar
        :param r_target: the target conversion of CO2 for whole process
        :param sp_paras: separator paras
        :param sf_target: separation ratio of each body
        :return: product in each stage
        """

        r_t = 0
        products, r_each = [], []
        reactor_feed = feed_comp
        sep_work, dHrs, Ws = [], [], []
        qs = []
        qconds = []
        prods_CH4O = []
        while r_t < r_target:
            # print(reactor_feed.values)
            product, _ = self.min_eq(T, P, reactor_feed)
            # metric of single reactor
            dHr = self.cal_H(T, P, product) - self.cal_H(T, P, feed_comp.values)  # dH during reaction, kW
            r = (reactor_feed[0] - product[0]) / reactor_feed[0]  # CO2 conversion
            r_each.append([r])

            # generate the feed for the next stage through separator
            sp_feed_paras = [T, P, product]
            reactor_feed, Q_diff, prod_CH4O = self.cond_sep(sp_feed_paras, sp_paras)
            # self.cond_sep(sp_feed_paras, sp_paras)
            # self.cond_sep(sp_feed_paras, sp_paras) self.cond_sep_along(sp_feed_paras, sp_paras, sf_target)
            # metric of separator
            qs.append(Q_diff)  # kW
            prods_CH4O.append(prod_CH4O)

            # metric of the whole process
            r_t = (feed_comp[0] - product[0]) / feed_comp[0]  # total conversion of CO2
            s_t = (feed_comp[0] - product[0] - product[-1]) / (feed_comp[0] - product[0])  # selectivity of CH4O
            products.append(product.tolist())
            dHrs.append(dHr)
        prods_CH4O = np.array(prods_CH4O)
        qs = np.array(qs)
        r_qin_qs = 1 - abs((r_t * feed_comp[0] * 50) / np.sum(qs))
        sim_metric = pd.Series([r_t, s_t, np.sum(qs), np.sum(prods_CH4O), r_qin_qs],
                               index=['r', 's', 'Q', 'Y_CH4O', 'rq_in_diff'])

        return sim_metric  # /produced_CH4O  # , np.array(qconds)

    def cond_sep(self, feed_paras, sep_paras):
        """
        condenser separator
        :param feed_paras: [T K, P bar, feed_flow mol/s]
        :param sep_paras: separator paras location=0, Din=Din, thick=Dd, Tc=Tc, heater=0
        :return:
        """
        feed_comp = feed_paras[2]  # mol/s
        # print(feed_comp)
        reactor_feed = pd.Series(feed_comp.copy(), index=self.comp)
        q, n = self.diffusion(feed_paras, **sep_paras)  # W/m mol/s m
        r_H2O_CH3OH = n[3] / n[2]
        # print(r_H2O_CH3OH, n)
        r_h_CH3OH = q / n[2]  # J/mol
        N_CH3OH = feed_comp[3] / r_H2O_CH3OH  # separated CH3OH, mol/s
        Q = r_h_CH3OH * N_CH3OH / 1000  # diffused heat, kW

        reactor_feed[3] = 0
        reactor_feed[2] = reactor_feed[2] - N_CH3OH
        # print(reactor_feed.values)

        if reactor_feed[2] < 0:
            raise ValueError('Gas flow should be positive!')
        else:
            return reactor_feed, Q, N_CH3OH

    def cond_sep_along(self, feed_paras, sep_paras, r_target):
        """
        condenser separator
        :param feed_paras: [T K, P bar, feed_flow mol/s]
        :param sep_paras: separator paras location=0, Din=Din, thick=Dd, Tc=Tc, heater=0
        :return:
        """
        feed_comp = feed_paras[2]  # mol/s
        reactor_feed = pd.Series(feed_comp.copy(), index=self.comp)
        q, n = self.diffusion(feed_paras, **sep_paras)  # W/m mol/s m
        L0 = feed_comp[2] * 0.9 / n[2]  # separated CH3OH, mol/s
        r_sim_CH3OH, r_sim_H2O = 0, 0
        L = abs(L0)
        while r_sim_CH3OH < r_target and r_sim_H2O <= 1:
            res_along = self.diff_along(feed_paras, L, **sep_paras)
            # print(res_along[:, -1])
            r_sim_CH3OH = (res_along[3, 0] - res_along[3, -1]) / res_along[3, 0]  # separation ratio of CH3OH
            r_sim_H2O = (res_along[4, 0] - res_along[4, -1]) / res_along[4, 0]  # separation ratio of H2O
            Q_diff = res_along[-2, -1] / 1000  # kW

            if r_sim_CH3OH < 0.6:
                L += 0.5
            elif 0.6 < r_sim_CH3OH < 0.8:
                L += 0.4
            elif 0.8 < r_sim_CH3OH < 0.9:
                L += 0.3
            else:
                L += 0.1
        reactor_feed.loc[:] = res_along[1:-2, -1]
        return reactor_feed, Q_diff

    def sep(self, Fin, Tin, Pin, Tos, Pos, sf=1, Fos=None):
        """
        calculate the metric of separator
        F1 is split into F2 and F3
        :param F1: molar flow rate (pd.Series), mol/s
        :param T: K
        :param P: bar
        :return:
        """
        if Fos is None:
            Fo_prod = pd.Series(0, index=Fin.index)
            Fo_prod.loc['Methanol'] = sf * Fin.loc['Methanol']
            Fo_prod.loc['H2O'] = min(sf * 1.25, 1) * Fin.loc['H2O']
            Fo_feed = Fin - Fo_prod
        Ws = self.cal_G(Tin, Pin, Fin.values) - \
             self.cal_G(Tos[0], Pos[0], Fo_feed.values) - self.cal_G(Tos[1], Pos[1], Fo_prod.values)
        Win = self.cal_G(Tin, Pin, Fin.values) - \
              self.cal_G(Tos[0], Pos[0], Fo_feed.values) - self.cal_G(Tos[1], Pos[1], Fo_prod.values)

    @staticmethod
    def diffusion(feed, location, Din, thick, Tc, heater):
        """

        :param feed: [T0, P0, F0: mol/s(ndarray)]
        :param location:
        :param Din:
        :param thick:
        :param Tc:
        :param heater:
        :return:
        """

        [T0, P0, F0] = feed
        Do = Din + thick * 2
        # property_feed = mixture_property(T0, xi_gas=pd.Series(F0 / np.sum(F0), index=subs), Pt=P0)
        insula_sim = Insulation(Do, Din, 1, location)
        res = insula_sim.flux(T0, P0, F0, Tc)  # mol/(s m) W/m
        h_diff, m_diff = res['hflux'], res['mflux']
        # r_h_m = h_diff / 1e3 / (m_diff[2])  # kJ/mol CH4O
        return h_diff, m_diff  # W/m, mol/(s m)

    @staticmethod
    def diff_along(feed, L, location, Din, thick, Tc, heater):

        [T0, P0, F0] = feed
        Do = Din + thick * 2
        insula_sim = Insulation(Do, Din, 1, location)

        def model(z, y):
            # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO
            # Tr]
            F = np.array(y[:5])
            Tr = y[-1]
            res_diff = insula_sim.flux(Tr, P0, F, Tc)  # mol/(s m) W/m
            dF_dz = res_diff["mflux"]
            dq_dz = res_diff['hflux']
            dTr_dz = 0  # res["Tvar"] + q_heater / heat_cap  # res_react['tc']
            res_dz = np.hstack((dF_dz, np.array([dq_dz, dTr_dz])))
            return res_dz

        # property_feed = mixture_property(T0, xi_gas=pd.Series(F0 / np.sum(F0), index=subs), Pt=P0)
        z_span = [0, L]
        ic = np.hstack((F0, np.array([0, T0])))
        res_sim = solve_ivp(model, z_span, ic, method='BDF', t_eval=np.linspace(0, L, 1000))  # LSODA BDF
        res = np.vstack((np.linspace(0, L, 1000), res_sim.y))
        return res


def find_best_cond(feed, T_range, Din_range, Dd_range):
    """
    find the lowest energy consumption with different conditions
    :param feed: molar flow rate (pd.Series), mol/s
    :param T_range:
    :param Din_range:
    :param Dd_range:
    :return:
    """
    sims_res = pd.DataFrame(columns=['Tc', 'Din', 'Dd', 'r', 's', 'Q', 'rq_in_diff'],
                            index=np.zeros(len(T_range) * len(Din_range) * len(Dd_range)))

    i = 0
    for Tc in T_range:
        for Din in Din_range:
            for Dd in Dd_range:
                diffusor_para = dict(location=0, Din=Din, thick=Dd, Tc=Tc, heater=0)
                gibbs_cal = Gibbs(in_gas.index)
                res = gibbs_cal.series_reactor(feed, 503, 70, r_target=0.95, sp_paras=diffusor_para)
                sims_res.iloc[i] = np.hstack((np.array([Tc, Din, Dd]), res))
                print(sims_res.iloc[i])
                i += 1
    path = f'res_Gibbs/eq_diff_{min(T_range)}_{max(T_range)}_' \
           f'{min(Din_range):.2f}_{max(Din_range):.2f}_' \
           f'{min(Dd_range):.3f}_{max(Dd_range):.3f}.xlsx'
    sims_res.to_excel(path, index=True, header=True, sheet_name='conversion')


def metric_single(feed, Tc, Din, Dd, r):
    diffusor_para = dict(location=0, Din=Din, thick=Dd, Tc=Tc, heater=0)
    gibbs_cal = Gibbs(feed.index)
    res = gibbs_cal.series_reactor(feed, 503, 70, r_target=r, sp_paras=diffusor_para, sf_target=0.85)
    return res


if __name__ == '__main__':
    in_gas = pd.Series([0.008154456, 0.024463369, 0, 0, 0], index=["CO2", "H2", "Methanol", "H2O", "carbon monoxide"])
    # 0.008154456, 0.024463369, 0, 0, 0
    # 0.00522348 0.01617213 0.00268013 0.00293098 0.00025085
    Tcs = np.arange(323, 403, 5)
    Dins = np.arange(0.02, 0.08, 0.01)
    thick = np.arange(0.002, 0.014, 0.002)

    find_best_cond(in_gas, Tcs, Dins, thick)
    # sim_res = metric_single(in_gas, Tc=353, Din=0.08, Dd=0.01, r=0.95)
    # print(sim_res)

    # metric_single()
