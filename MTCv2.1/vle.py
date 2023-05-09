import time
import warnings
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
import scipy.optimize as opt

warnings.filterwarnings('ignore')
# T_gas = 200  # 310.93  # K
# P_gas = 30  # 6.3  # bar
R = 8.314


class VLE:
    def __init__(self, T, comp):
        self.T = T
        self.mix = comp / comp.sum()

        self.condensate = comp[['Methanol', 'H2O']] / comp[['Methanol', 'H2O']].sum()
        self.num = len(self.mix.index)
        self.index = pd.Series(np.arange(self.num), index=self.mix.index)

        # read properties from coolprop
        [self.Tc, self.Pc, self.Psat, self.Omega] = np.zeros((4, self.num))
        for i in range(self.num):
            self.Tc[i] = PropsSI('Tcrit', comp.index[i])
            self.Pc[i] = PropsSI('Pcrit', comp.index[i]) * 1e-5
            self.Psat[i] = PropsSI('P', 'T', self.T, 'Q', 1, comp.index[i]) * 1e-5 if self.T < self.Tc[i] else 1e5
            self.Omega[i] = PropsSI('acentric', comp.index[i])
        # calculate para in Eos for each component
        alpha = (1 + (0.48 + 1.574 * self.Omega - 0.176 * self.Omega ** 2)
                 * (1 - (self.T / self.Tc) ** 0.5)) ** 2
        self.a = 0.42748 * (R * 10) ** 2 * self.Tc ** 2 * alpha / self.Pc
        self.b = 0.08664 * (R * 10) * self.Tc / self.Pc

        # mixing rule parameter CO2 H2 CH3OH H2O CO
        self.k_a = np.array([[0, -0.3462, 0.0148, 0.0737, 0],
                             [-0.3462, 0, 0, 0, 0.0804],
                             [0.0148, 0, 0, -0.0789, 0],
                             [0.0737, 0, -0.0789, 0, 0],
                             [0, 0.0804, 0, 0, 0]])

    @staticmethod
    def para_mix(comp, a, b, kij):
        num = len(comp)
        # calculate the mix para in EoS
        b_mix = np.sum(b * comp)
        a_mix = 0
        for m in range(num):
            for n in range(num):
                a_mix += comp[m] * comp[n] * (a[m] * a[n]) ** 0.5 * (1 - kij[m, n])
        return a_mix, b_mix

    @staticmethod
    def func_z(beta, q, status):
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
        index = self.index[comp.index]
        num = len(index)
        a, b = self.a[index], self.b[index]
        k_mix = self.k_a[index][:, index]
        [q_ba, a_ba] = np.zeros((2, num))

        # the mix para in EoS for vapor phase
        a_mix, b_mix = self.para_mix(comp, a, b, k_mix)
        beta_mix = b_mix * P / (R * 10) / self.T
        q_mix = a_mix / b_mix / (R * 10) / self.T
        Z_guess = 1e-5 if phase == 1 and num == 2 else 0.5
        Z_mix = opt.fsolve(self.func_z(beta_mix, q_mix, phase), [Z_guess])[0]
        Z_mix = 1e-5 if Z_mix < 0 else Z_mix
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
        return phi

    def dew_p(self, spec, x_guess=None):
        y = self.mix.iloc[spec] / self.mix.iloc[spec].sum()
        num = len(y)
        comp = pd.DataFrame(index=['V', 'L'], columns=y.index)
        comp.iloc[0] = y.values

        # find the equilibrium pressure and mol fraction of liquid phase
        comp.iloc[1] = x_guess if x_guess is not None else y.values
        P_min = np.min(self.Psat) if len(spec) == 2 else 20
        for P in np.arange(P_min, 80, 0.05):
            delta_K_sum = 1e5
            K_sum_K_pre = 10
            while delta_K_sum > 0.01:
                phi = np.empty((2, num))
                for i in range(2):
                    # cycle for each phase
                    phi[i] = self.phi(comp.iloc[i], P, phase=i)
                    # phi[i] = fi[i] / P / comp.iloc[i].values
                K = phi[1] / phi[0]
                if np.isnan(np.sum(K)): return None
                K_sum_cal = np.sum(comp.iloc[0].values / K)
                comp.iloc[1] = comp.iloc[0] / K / K_sum_cal
                delta_K_sum = abs(K_sum_cal - K_sum_K_pre)
                K_sum_K_pre = K_sum_cal
            if abs(K_sum_cal - 1) < 0.005:
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
                return res
            elif K_sum_cal > 1:
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
                return res

    @property
    def dew_p_all(self):
        dew_p_cds = self.dew_p([2, 3])
        x_guess = np.ones(self.num) * 0.1
        x_guess[2:4] = (1 - 0.01 * (self.num - 2)) * dew_p_cds['comp'].iloc[1]
        res = self.dew_p(np.arange(self.num), x_guess)
        return res

    def flash(self, P):
        comp = pd.DataFrame(index=['V', 'L'], columns=self.mix.index)
        fi = np.zeros((2, self.num))
        fi[1] = 1
        K = np.exp(5.37 * (1 + self.Omega) * (1 - 1 / (self.T / self.Tc))) / (P / self.Pc)
        m = 0
        while np.sum(abs(fi[0] - fi[1])) > 1e-5:
            vol = opt.fsolve(self.func_l(self.mix.values, K), 1e-3)[0]
            comp.iloc[1] = self.mix.values / (vol + (1 - vol) * K)
            comp.iloc[0] = K * comp.iloc[1]
            fi[0] = self.phi(comp.iloc[0], P, 0) * P * comp.iloc[0]
            fi[1] = self.phi(comp.iloc[1], P, 1) * P * comp.iloc[1]
            K = fi[1] * K / fi[0]
            m += 1
        return comp


# # exp = [0.209330615, 0.670652596, 0.028540247, 0.043940264]
# # ['CO2', 'H2', 'Methanol', 'H2O'] ['CO2', 'H2', 'Methanol', 'H2O', 'CO'] 'CO2', 'Methanol', 'H2O']
# a = time.time()
# exp = [0.243889,0.731673,0.000021,0.000023,0.024394]
# exp = [0.240200, 0.722865, 0.005017, 0.006150, 0.025768]
# # # #[0.209059763, 0.669913119, 0.029543594, 0.043759505, 0.047724019]
# mix = pd.Series(exp, index=['CO2', 'H2', 'Methanol', 'H2O', 'CO'])  # 'Methane', 'Butane' 'N2', 'Methane'
# aa = VLE(T=333.15, comp=mix)
# # # #
# print(aa.flash(70))
# print(aa.dew_p_all)
# b=time.time()
# print(b-a)

a = [9.761762132,34.27631344,1.566465771,1.71562863,2.679830027]
b = np.array([0.828404546,2.908763139,0.132933721,0.145592009,0.227416254])

mix = pd.Series(b, index=['CO2', 'H2', 'Methanol', 'H2O', 'CO'])

aa = VLE(T=318, comp=mix)
print(aa.flash(P=50))