import time
import warnings
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
import scipy.optimize as opt

from thermo import (ChemicalConstantsPackage, SRKMIX, FlashVL, CEOSLiquid, CEOSGas, HeatCapacityGas,
                    GibbsExcessLiquid, MSRKMIX, SRKMIXTranslated)

warnings.filterwarnings('ignore')
# T_gas = 200  # 310.93  # K
# P_gas = 30  # 6.3  # bar
R = 8.314


def mixture_property(T, xi_gas, Pt, z=1, rho_only=False):
    """
    calculate the properties of gas mixture
    :param T: gas temperature, K
    :param xi_gas: molar fraction; pd.Serize
    :param Pt: total pressure, bar
    :param z: compression factor
    :return: thermal conductivity W/(m K), viscosity Pa s, heat capacity J/mol/K; pd.series
    """
    # prepare data for calculation
    n = len(xi_gas.index)  # number of gas species

    [cp, k, vis, M, rho] = np.ones((5, n)) * 1e-5
    pi_gas = xi_gas * Pt * 1e5  # convert bar to pa
    Ti_sat = pd.Series(np.ones(n) * 100, index=xi_gas.index)
    if 'Methanol' in xi_gas.index:
        try:
            Ti_sat['Methanol'] = PropsSI('T', 'P', pi_gas['Methanol'], 'Q', 1, 'Methanol')
        except ValueError:
            Ti_sat['Methanol'] = 300
    if "H2O" in xi_gas.index:
        try:
            Ti_sat['H2O'] = PropsSI('T', 'P', pi_gas['H2O'], 'Q', 1, 'H2O')
        except ValueError:
            Ti_sat['H2O'] = 300
    i = 0
    for comp in xi_gas.index:
        M[i] = PropsSI('MOLARMASS', 'T', T, 'P', 1e5, comp)  # molar weight, g/mol
        i += 1
    M_m = np.sum(M * xi_gas)  # molar weight of mixture
    rho_m = Pt * 1E5 * M_m / 1000 / (z * R * T)  # kg/m3 np.sum(rho)
    if rho_only:
        return pd.Series([0, 0, rho_m, cp[2], cp[3], 0],
                         index=["k", "vis", 'rho', 'cp_' + xi_gas.index[2], 'cp_' + xi_gas.index[3], "cp_m"])

    i = 0
    # calculate the properties of pure gases
    for comp in xi_gas.index:
        gas = "N2" if comp == "CO" else comp  # "CO" is not available in CoolProp
        if pi_gas[comp] > 1000:
            if T > Ti_sat[comp] * 1.05:
                # thermal conductivity, W/(m K)
                k[i] = PropsSI('L', 'T', T, 'P', Pt, gas)
                # viscosity, Pa S
                vis[i] = PropsSI('V', 'T', T, 'P', Pt, gas)
                # heat capacity, J/(mol K)
                cp[i] = PropsSI('CPMOLAR', 'T', T, 'P', Pt, gas)
                # density, kg/m3
                # rho[i] = PropsSI('D', 'T', T, 'P', xi_gas[comp], gas)
            else:
                cp[i] = PropsSI('CPMOLAR', 'T', T, 'Q', 1, gas)
                k[i] = PropsSI('L', 'T', T, 'Q', 1, gas)
                vis[i] = PropsSI('V', 'T', T, 'Q', 1, gas)
                # rho[i] = PropsSI('D', 'T', T, 'Q', 1, gas)
        else:
            # thermal conductivity, W/(m K)
            k[i] = 0
            # viscosity, Pa S
            vis[i] = 1e-10
            # heat capacity, J/(mol K)
            cp[i] = 0
            # density, kg/m3
            rho[i] = 0
        i += 1

    # calculate the properties of mixture
    cp_m = np.sum(cp * xi_gas)
    phi, denominator = np.ones((n, n)), np.ones((n, n))  # Wilke coefficient
    vis_m, k_m = 0, 0
    for i in range(n):
        for j in np.arange(n):
            phi[i, j] = (1 + (vis[i] / vis[j]) ** 0.5 * (M[j] / M[i]) ** 0.25) ** 2 / (8 * (1 + M[i] / M[j])) ** 0.5
            denominator[i, j] = xi_gas[j] * phi[i, j]  # if i != j else 0
        vis_m += xi_gas[i] * vis[i] / np.sum(denominator[i])
        k_m += xi_gas[i] * k[i] / np.sum(denominator[i])
    return pd.Series([k_m, vis_m, rho_m, cp[2], cp[3], cp_m],
                     index=["k", "vis", 'rho', 'cp_' + xi_gas.index[2], 'cp_' + xi_gas.index[3], "cp_m"])


class VLE:
    """
    calculate the VLE properties including:
    fugacity, fugacity coe, dew point of mixture, flash calculation
    """

    def __init__(self, T, comp):
        """
        Initialize the VLE object.
        :param T: Temperature (K)
        :param comp: component of mix, name in list
        """
        self.T = T
        self.comp = comp
        self.num = len(self.comp)
        self.index = pd.Series(np.arange(self.num), index=self.comp)
        self.Tc, self.Pc, self.Psat, self.Omega = self._calculate_properties()
        self.a, self.b = self._calculate_parameters()
        self.k_a = self._mix_rule()

    def _calculate_properties(self):
        """
        Calculate critical properties and Psat for each component.
        """
        Tc = np.zeros(self.num)
        Pc = np.zeros(self.num)
        Psat = np.zeros(self.num)
        Omega = np.zeros(self.num)

        for i in range(self.num):
            Tc[i] = PropsSI('Tcrit', self.comp[i])
            Pc[i] = PropsSI('Pcrit', self.comp[i]) * 1e-5
            Psat[i] = PropsSI('P', 'T', self.T, 'Q', 1, self.comp[i]) * 1e-5 if self.T < Tc[i] else 1e5
            Omega[i] = PropsSI('acentric', self.comp[i])

        return Tc, Pc, Psat, Omega

    def _calculate_parameters(self):
        """
        Calculate a and b parameters for each component.
        """
        alpha = (1 + (0.48 + 1.574 * self.Omega - 0.176 * self.Omega ** 2)
                 * (1 - (self.T / self.Tc) ** 0.5)) ** 2
        a = 0.42748 * (R * 10) ** 2 * self.Tc ** 2 * alpha / self.Pc
        b = 0.08664 * (R * 10) * self.Tc / self.Pc
        return a, b

    def _mix_rule(self):
        """
        define the mixing rule parameter
        """
        if self.comp == ["CO2", "H2", "Methanol", "H2O", "CO"]:
            # mixing rule parameter CO2 H2 CH3OH H2O CO
            k = np.array([[0, -0.3462, 0.0148, 0.0737, 0],
                          [-0.3462, 0, 0, 0, 0.0804],
                          [0.0148, 0, 0, -0.0789, 0],
                          [0.0737, 0, -0.0789, 0, 0],
                          [0, 0.0804, 0, 0, 0]])
            # k = np.array([[0, -0.0007, 0.1, 0.3, 0.1164],
            #               [0.1164, 0, -0.125, -0.745, -0.0007],
            #               [0.1, -0.125, 0, -0.075, -0.37],
            #               [0.3, -0.745, -0.075, 0, -0.474],
            #               [0.1164, -0.0007, -0.37, -0.474, 0]])
        else:
            k = np.zeros((self.num, self.num))
        return k

    @staticmethod
    def para_mix(comp, a, b, kij):
        """
        Calculate the mixing parameters for EoS.
        :param comp: Molar fraction of fluid (numpy array)
        :param a: a parameters (numpy array)
        :param b: b parameters (numpy array)
        :param kij: Binary interaction coefficients (numpy array)
        :return: a_mix, b_mix (floats)
        """
        num = len(comp)
        b_mix = np.sum(b * comp)
        a_mix = 0
        for m in range(num):
            for n in range(num):
                a_mix += comp[m] * comp[n] * (a[m] * a[n]) ** 0.5 * (1 - kij[m, n])
        return a_mix, b_mix

    @staticmethod
    def func_z(beta, q, status):
        # phase 1 refers to liquid phase, 0 refers to vapor phase
        if status == 1:
            def func(x):
                return beta + x * (x + beta) * (1 + beta - x) / q / beta - x

            return func
        elif status == 0:
            def func(x):
                return 1 + beta - q * beta * (x - beta) / x / (x + beta) - x

            return func

    @staticmethod
    def func_v(z, K):
        def func(x):
            return np.sum(z * (K - 1) / (1 + x * (K - 1)))

        return func

    @staticmethod
    def func_l(z, K):
        def func(x):
            return np.sum(z * (1 - K) / (x + (1 - x) * K))

        return func

    def phi(self, comp, P, phase=0):
        """
        Calculate fugacity coefficients (phi) for the components in a mixture.
        :param comp: Molar fraction of fluid (pandas Series)
        :param P: Total pressure, bar
        :param phase: Phase (0 for vapor, 1 for liquid)
        :return: Fugacity coefficients (numpy array), compression factor of mixture
        """
        # extract the fluid parameters
        index_list = comp.index.tolist()
        index = self.index.loc[index_list]
        num = len(index)
        a, b = self.a[index], self.b[index]
        k_mix = self.k_a[index][:, index]
        [q_ba, a_ba] = np.zeros((2, num))

        # the mix para in EoS for vapor phase
        a_mix, b_mix = self.para_mix(comp, a, b, k_mix)
        beta_mix = b_mix * P / (R * 10) / self.T

        q_mix = a_mix / b_mix / (R * 10) / self.T
        Z_guess = 1e-3 if phase == 1 else 0.8
        Z_mix = opt.fsolve(self.func_z(beta_mix, q_mix, phase), [Z_guess])[0]
        Z_mix = 1e-3 if Z_mix < 0 else Z_mix
        I_mix = np.log((Z_mix + beta_mix) / Z_mix)
        ln_phi = np.empty(num)
        for j in range(num):
            # cycle for each component
            a_ba[j] = 0
            for m in range(num):
                a_ba[j] += (a[j] * a[m]) ** 0.5 * (1 - k_mix[j, m]) * comp[m] * 2
            a_ba[j] -= a_mix
            q_ba[j] = q_mix * (1 + a_ba[j] / a_mix - b[j] / b_mix)
            ln_phi[j] = b[j] * (Z_mix - 1) / b_mix - np.log(Z_mix - beta_mix) - q_ba[j] * I_mix
        phi = np.exp(ln_phi)
        return phi, Z_mix

    def dew_p(self, y, x_guess=None):
        """
        Calculate dew point pressure for a given component or set of components and initial liquid phase composition.

        :param y: Molar fraction of equilibrium vapor (pandas Series)
        :param x_guess: Initial liquid phase composition
        :return: Dew point pressure and composition (dict)
        """
        num = len(y)
        comp = pd.DataFrame(index=['V', 'L1'], columns=y.index)
        comp.iloc[0] = y.values

        # find the equilibrium pressure and mol fraction of liquid phase
        comp.iloc[1] = x_guess if x_guess is not None else self.Psat / np.sum(self.Psat)  # y.values
        P_min = np.min(self.Psat) if len(y) == 2 else 48  # when only CH3OH\H2O is considered, P_min decided by Psat
        step = 0.01 if (len(y) == 2 and self.T < 373) else 1
        for P in np.arange(P_min, 200, step):
            delta_K_sum = 1e5
            K_sum_K_pre = 10
            while delta_K_sum > 0.05:
                phi = np.empty((2, num))
                for i in range(2):
                    # cycle for each phase
                    phi[i], _ = self.phi(comp.iloc[i], P, phase=i)
                K = phi[1] / phi[0]
                if np.isnan(np.sum(K)): return None
                K_sum_cal = np.sum(comp.iloc[0].values / K)
                comp.iloc[1] = comp.iloc[0] / K / K_sum_cal
                delta_K_sum = abs(K_sum_cal - K_sum_K_pre)
                K_sum_K_pre = K_sum_cal

            if abs(K_sum_cal - 1) < 0.01:
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
                return res
            elif K_sum_cal > 1:
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
                return res

    def flash(self, P, mix):
        """
        Perform flash calculations to find the phase equilibrium composition at a specified pressure.

        :param P: Pressure
        :param mix: Molar fraction of mix fluid (pandas Series)
        :return: Phase equilibrium composition (DataFrame)
        """
        comp = pd.DataFrame(index=['V', 'L1'], columns=mix.index)
        fi = np.zeros((2, self.num))
        fi[1] = 1
        K = np.exp(5.37 * (1 + self.Omega) * (1 - 1 / (self.T / self.Tc))) / (P / self.Pc)
        m, dev = 0, 1
        tol, max_iter, vol_guess = 1e-3, 50, [1e-3, 5e-3, 1e-2, 0.05, 0.1]
        i = 0
        while dev > tol:
            vol = opt.fsolve(self.func_l(mix.values, K), vol_guess[i])[0]
            comp.iloc[1] = mix.values / (vol + (1 - vol) * K)
            comp.iloc[0] = K * comp.iloc[1]
            phi, _ = self.phi(comp.iloc[0], P, 0)
            fi[0] = phi * P * comp.iloc[0]
            phi, _ = self.phi(comp.iloc[1], P, 1)
            fi[1] = phi * P * comp.iloc[1]
            K = fi[1] * K / fi[0]
            dev = np.max(abs(fi[0] - fi[1]) / fi[0])
            if m % max_iter == 0 and m != 0:
                print(abs(fi[0] - fi[1]) / fi[0])
                print("reach max iteration")
                i += 1
                # return comp, phi
            m += 1
        return comp, phi


class Thermo:
    """
    calculate thermo-physical properties including
    formation of enthalpy, formation of Gibbs energy
    at given comps, temperature
    """

    def __init__(self, hf_para=None, gf_para=None):

        # fit para for calculate enthalpy of formation, and Gibbs energy of formation
        self.gf_paras = self.__Gf_para() if gf_para is None else gf_para
        self.hf_paras = self.__Hf_para() if hf_para is None else hf_para
        self.gf_fit, self.hf_fit = self.fit_para

    @staticmethod
    def __Gf_para():
        # # Parameters for calculating gibbs energy of formation
        # # CO2 H2 CH3OH H2O CO
        Ts = np.arange(300, 1100, 100)
        Gfs = np.array([[-394.379, -394.656, -394.914, -395.152, -395.367, -395.558, -395.724, -395.865],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [-162.057, -148.509, -134.109, -119.125, -103.737, -88.063, -72.188, -56.170],
                        [-228.5, -223.9, -219.05, -214.008, -208.814, -203.501, -198.091, -192.603],
                        [-137.333, -146.341, -155.412, -164.480, -173.513, -182.494, -191.417, -200.281]])

        Gfs = pd.DataFrame(Gfs, index=["CO2", "H2", "Methanol", "H2O", "CO"], columns=Ts)
        return Gfs

    @staticmethod
    def __Hf_para():
        # # Parameters for calculating enthalpy of formation
        # # CO2 H2 CH3OH H2O CO
        Ts = np.arange(300, 1100, 100)

        Hfs = np.array([[-393.511, -393.586, -393.672, -393.791, -393.946, -394.133, -394.343, -394.568],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [-201.068, -204.622, -207.750, -210.387, -212.570, -214.350, -215.782, -216.916],
                        [-241.844, -242.845, -243.822, -244.751, -245.620, -246.424, -247.158, -247.820],
                        [-110.519, -110.121, -110.027, -110.157, -110.453, -110.870, -111.378, -111.952]])
        Hfs = pd.DataFrame(Hfs, index=["CO2", "H2", "Methanol", "H2O", "CO"], columns=Ts)
        return Hfs

    @property
    def fit_para(self):
        """
        fit para for calculating the energy (Gibbs energy or enthalpy) of formation for a species
        """
        gf_fit_paras = {}
        hf_fit_paras = {}
        for comp in self.gf_paras.index:
            gf_fit_paras[comp] = np.polyfit(self.gf_paras.loc[comp].index, self.gf_paras.loc[comp].values, 2)
            hf_fit_paras[comp] = np.polyfit(self.hf_paras.loc[comp].index, self.hf_paras.loc[comp].values, 2)
        return gf_fit_paras, hf_fit_paras

    def sat(self, pi):
        """
        :param pi: partial pressure (pd.Series)
        :return:
        """
        pc = pd.Series(index=pi.index)
        for comp in pi.index:
            pc = PropsSI('Pcrit', comp) * 1e-5

    def H(self, T, comps):
        """
        calculate formation of enthalpy at T for given comp
        :param T: K
        :param comps: list
        :return: molar enthalpy (pd.Series)
        """
        H = pd.Series(index=comps)
        for comp in comps:
            H.loc[comp] = np.polyval(self.hf_fit[comp], T)
        return H

    def H1(self, T, pi):
        """
        calculate formation of enthalpy at T for given comp
        :param T: K
        :param pi: partial pressure (pd.Series)
        :return: molar enthalpy (pd.Series)
        """
        H = pd.Series(index=pi.index)
        Tc = pd.Series(index=pi.index)  # critical temperature
        ps = pd.Series(index=pi.index)  # saturated pressure
        for comp in pi.index:
            Tc[comp] = PropsSI('Pcrit', comp)
            ps[comp] = PropsSI('P', 'T', T, 'Q', 1, comp) * 1e-5 if T < Tc[comp] else np.nan
            if not np.isnan(ps[comp]) and pi[comp] > ps[comp]:
                # liquid phase
                h_vl = PropsSI('H', 'T', T, 'Q', 1, comp) - PropsSI('H', 'T', T, 'Q', 0, comp)
                # v_ps, v_pi =
            else:
                H.loc[comp] = np.polyval(self.hf_fit[comp], T)
        return H

    def G(self, T, pi):
        """
        calculate formation of gibbs energy at T for given partial pressure
        :param T: K
        :param pi: partial pressure (pd.Series)
        :return: molar gibbs energy (pd.Series)
        """
        G = pd.Series(index=pi.index)
        for comp in pi.index:
            if pi[comp] < 1e-15:
                G.loc[comp] = 0
            else:
                G.loc[comp] = np.polyval(self.gf_fit[comp], T) + 8.314 * T * np.log(pi[comp] / 1) / 1000
        return G


class VLEThermo:
    def __init__(self, comp):
        self.comp = comp.copy()
        self.comp[comp == 'CO'] = 'carbon monoxide'
        self.const, self.cor = ChemicalConstantsPackage.from_IDs(self.comp)
        self.cp, self.eos_kw = self.__eos_paras()

    def __eos_paras(self, eos='SRK'):
        n_count = len(self.comp)
        cp_cal = [HeatCapacityGas(CASRN=self.const.CASs[i], MW=self.const.MWs[i], method='TRCIG') for i in
                  range(n_count)]
        kijs_msrk = np.array([[0, 0.1164, 0.1, 0.3, 0.1164],
                              [0.1164, 0, -0.125, -0.745, -0.0007],
                              [0.1, -0.125, 0, -0.075, -0.37],
                              [0.3, -0.745, -0.075, 0, -0.474],
                              [0.1164, -0.0007, -0.37, -0.474, 0]])

        kijs_srk = np.array([[0, -0.3462, 0.0148, 0.0737, 0],
                             [-0.3462, 0, 0, 0, 0.0804],
                             [0.0148, 0, 0, -0.0789, 0],
                             [0.0737, 0, -0.0789, 0, 0],
                             [0, 0.0804, 0, 0, 0]])

        p = [0, 0, 0.2359, 0.1277, 0]
        eos_kwargs_srk = dict(Tcs=np.array(self.const.Tcs), Pcs=np.array(self.const.Pcs),
                              omegas=np.array(self.const.omegas), kijs=kijs_srk)
        eos_kwargs_msrk = dict(Tcs=np.array(self.const.Tcs), Pcs=np.array(self.const.Pcs),
                               omegas=np.array(self.const.omegas), kijs=kijs_msrk, S2s=p)
        eos_kwargs = eos_kwargs_srk if eos == 'SRK' else eos_kwargs_msrk
        return cp_cal, eos_kwargs

    def p_dew(self, T, x):
        frac = x / np.sum(x)
        gas = CEOSGas(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw)
        liquid = CEOSLiquid(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw)
        flasher = FlashVL(self.const, self.cor, liquid=liquid, gas=gas)
        try:
            return flasher.flash(zs=frac, T=T, VF=1).P / 1E5
        except UnboundLocalError:
            return 15e5

    def p_bub(self, T, x):
        frac = x / np.sum(x)
        gas = CEOSGas(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw)
        liquid = CEOSLiquid(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw)
        flasher = FlashVL(self.const, self.cor, liquid=liquid, gas=gas)
        try:
            return flasher.flash(zs=frac, T=T, VF=0).P / 1E5
        except UnboundLocalError:
            return 15e5

    def phi(self, T, P, x):
        frac = x / np.sum(x)
        gas = CEOSGas(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw, T=T, P=P*1e5,zs=frac)
        liquid = CEOSLiquid(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw)
        flasher = FlashVL(self.const, self.cor, liquid=liquid, gas=gas)
        TP = flasher.flash(zs=frac, T=T, P=P * 1E5)
        # print(frac)
        return TP.phis()

    def flash(self, T, P, x):
        frac = x / np.sum(x)
        gas = CEOSGas(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw)
        liquid = CEOSLiquid(SRKMIX, HeatCapacityGases=self.cp, eos_kwargs=self.eos_kw)
        flasher = FlashVL(self.const, self.cor, liquid=liquid, gas=gas)
        TP = flasher.flash(zs=frac, T=T, P=P * 1E5)

        # print(frac)
        return TP.gas.zs

# mixture_property(490, pd.Series([0.25, 0.75], index=['CO2', 'hydrogen']), 70)
# tt = VLE(293, ["H2O"])
# print(tt.phi(pd.Series([1], index=["H2O"]), 1, 1))
