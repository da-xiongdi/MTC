import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy.optimize import fsolve

T_gas = 200  # 310.93  # K
P_gas = 30  # 6.3  # bar
R = 8.314


class VLE:
    def __init__(self, T, comp):
        self.T = T
        self.mix = comp / comp.sum()
        self.condensate = comp[['Methanol', 'H2O']] / comp[['Methanol', 'H2O']].sum()
        self.num = len(self.mix.index)

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
        # self.k_a = self.k_a[[2, 3]][:, [2, 3]]

    def para_mix(self, comp):
        # calculate the mix para in EoS
        # a_multi = np.matmul(self.a.reshape(num, 1), self.a.reshape(1, num)) ** 0.5 \
        #           * np.array([[1, 1.0789], [1.0789, 1]])
        b_mix = np.sum(self.b * comp)
        a_mix = 0
        for i in range(self.num):
            for j in range(self.num):
                a_mix += comp[i] * comp[j] * (self.a[i] * self.a[j]) ** 0.5 * (1 - self.k_a[i, j])
        # a_mix = np.matmul(np.matmul(comp.reshape(1, num), a_multi), comp.reshape(num, 1))[0, 0]  # * (1 + 0.0789)
        return a_mix, b_mix

    def para_mix2(self, comp):
        # calculate the mix para in EoS

        num = len(comp)
        # CH3OH H2O
        alpha = np.array([[0, 0.15], [0.15, 0]])
        gamma = np.array([[0, 780.1], [-220, 0]])
        tau = gamma / self.T
        [numerator, denominator, a_ba] = np.zeros((3, num))
        for i in range(num):
            for j in range(num):
                numerator[i] += tau[j, i] * self.b[j] * comp[j] * np.exp(-alpha[j, i] * tau[j, i])
                denominator[i] += self.b[j] * comp[j] * np.exp(-alpha[j, i] * tau[j, i])
            a_ba[i] = self.b[i] * (self.a[i] / self.b[i] - (np.sum(numerator[i] / denominator[i]) +
                                                            np.sum(-comp * numerator[i] * self.b[i] / denominator[
                                                                i] ** 2)) * (R * 10) * self.T / np.log(2))
        factor = np.sum(comp * numerator / denominator)

        b_mix = np.sum(self.b * comp)
        a_mix = (np.sum(self.a * comp / self.b) - factor * (R * 10) * self.T / np.log(2)) * b_mix  # * (1 - 0.094)
        # print(a_mix)
        return a_mix, b_mix, a_ba

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

    @property
    def dew_p(self):
        comp = np.zeros((2, self.num))
        comp[0] = self.mix.values

        [a_mix, b_mix, Z_mix, I_mix] = np.zeros((4, 2))
        [q_ba, a_ba, phi] = np.zeros((3, 2, self.num))

        # the mix para in EoS for vapor phase
        # a_mix[0], b_mix[0], a_ba[0] = self.para_mix2(comp[0])
        a_mix[0], b_mix[0] = self.para_mix(comp[0])

        # find the equilibrium pressure and mol fraction of liquid phase
        comp[1] = np.array([0.01, 0.01, 0.2, 0.77, 0.01])  # self.mix.values
        for P in np.arange(np.min(self.Psat), 80, 0.01):
            delta_K_sum = 1e5
            K_sum_K_pre = 10
            while delta_K_sum > 0.01:
                ln_phi = np.empty((2, self.num))
                # a_mix[1], b_mix[1], a_ba[1] = self.para_mix2(comp[1])
                a_mix[1], b_mix[1] = self.para_mix(comp[1])
                beta_mix = b_mix * P / (R * 10) / self.T
                q_mix = a_mix / b_mix / (R * 10) / self.T
                for i in range(2):
                    # cycle for each phase
                    Z_guess = 1e-5 if i == 1 else 0.8
                    Z_mix[i] = fsolve(self.func_z(beta_mix[i], q_mix[i], status=i), [Z_guess])
                    I_mix[i] = np.log((Z_mix[i] + beta_mix[i]) / Z_mix[i])
                    for j in range(self.num):
                        # cycle for each component
                        a_ba[i, j] = 0
                        for m in range(self.num):
                            a_ba[i, j] += (self.a[j] * self.a[m]) ** 0.5 * (1 - self.k_a[j, m]) * comp[i, m] * 2
                        a_ba[i, j] -= a_mix[i]
                        # a_ba[i, j] = (np.sum((self.a[j] * self.a) ** 0.5 * comp[i] * k_mix[i,j]) * 2 - a_mix[i])
                        q_ba[i, j] = q_mix[i] * (1 + a_ba[i, j] / a_mix[i] - self.b[j] / b_mix[i])
                        if Z_mix[i] - beta_mix[i] >= 0:
                            ln_phi[i, j] = self.b[j] * (Z_mix[i] - 1) / b_mix[i] - \
                                           np.log(Z_mix[i] - beta_mix[i]) - q_ba[i, j] * I_mix[i]
                        else:
                            ln_phi[i, j] = 10 if j != 2 or 3 else 0.5
                phi = np.exp(ln_phi)
                K = phi[1] / phi[0]
                K_sum_cal = np.sum(comp[0] / K)
                comp[1] = comp[0] / K / K_sum_cal
                delta_K_sum = abs(K_sum_cal - K_sum_K_pre)
                K_sum_K_pre = K_sum_cal
            if abs(K_sum_cal - 1) < 0.005:
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
                return res
            elif K_sum_cal > 1:
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
                return res

    @property
    def dew_p2(self):
        comp = np.zeros((2, self.num))
        comp[0] = self.mix.values

        [a_mix, b_mix, Z_mix, I_mix] = np.zeros((4, 2))
        [q_ba, a_ba, phi, Z, I_sin] = np.zeros((5, 2, self.num))

        # the mix para in EoS for vapor phase
        a_mix[0], b_mix[0] = self.para_mix(comp[0])

        # find the equilibrium pressure and mol fraction of liquid phase
        comp[1] = self.mix.values
        for P in np.arange(np.min(self.Psat), 50, 0.01):
            beta = self.b * P / (R * 10) / self.T
            q = self.a / self.b / (R * 10) / self.T
            delta_K_sum = 1e5
            K_sum_pre = 1
            while delta_K_sum > 0.005:
                ln_phi = np.empty((2, self.num))
                a_mix[1], b_mix[1] = self.para_mix(comp[1])
                beta_mix = b_mix * P / (R * 10) / self.T
                q_mix = a_mix / b_mix / (R * 10) / self.T
                Z_guess = 0.8
                for j in range(self.num):
                    Z_guess = 0.8
                    Z[0, j] = fsolve(self.func_z(beta[j], q[j], status=0), [Z_guess])
                    I_sin[0, j] = np.log((Z[0, j] + beta[j]) / Z[0, j])

                Z_mix[0] = fsolve(self.func_z(beta[i], q_mix[i], status=0), [Z_guess])
                I_mix[0] = np.log((Z_mix[0] + beta_mix[0]) / Z_mix[0])

                for j in range(self.num):
                    q_ba[0, j] = (1 - Z[0,])
                for i in range(2):
                    # cycle for each phase
                    Z_guess = 1e-5 if i == 1 else 0.8
                    Z_mix[i] = fsolve(self.func_z(beta[i], q_mix[i], status=i), [Z_guess])
                    I_mix[i] = np.log((Z_mix[i] + beta_mix[i]) / Z_mix[i])
                    for j in range(self.num):
                        # cycle for each component
                        Z[i, j] = fsolve(self.func_z(beta[i], q_mix[i], status=i), [Z_guess])
                        a_ba[i, j] = np.sum((self.a[j] * self.a) ** 0.5 * comp[i]) * 2 - a_mix[i]
                        q_ba[i, j] = q_mix[i] * (1 + a_ba[i, j] / a_mix[i] - self.b[j] / b_mix[i])
                        if Z_mix[i] - beta_mix[i] >= 0:
                            ln_phi[i, j] = self.b[j] * (Z_mix[i] - 1) / b_mix[i] - \
                                           np.log(Z_mix[i] - beta_mix[i]) - q_ba[i, j] * I_mix[i]
                        else:
                            ln_phi[i, j] = -1e10
                phi = np.exp(ln_phi)
                K = phi[1] / phi[0]
                K_sum_cal = np.sum(comp[0] / K)
                comp[1] = comp[0] / K / K_sum_cal
                delta_K_sum = abs(K_sum_cal - K_sum_pre)
                K_sum_pre = K_sum_cal
            if abs(K_sum_cal - 1) < 0.01:
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
                return res
            elif abs(K_sum_cal - 1) < abs(K_sum_pre - 1):
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
        return res

    @property
    def dew_p3(self):
        comp = np.zeros((2, self.num))
        comp[0] = self.mix.values

        [a_mix, b_mix, Z_mix, I_mix] = np.zeros((4, 2))
        [q_ba, a_ba, phi] = np.zeros((3, 2, self.num))

        # the mix para in EoS for vapor phase
        a_mix[0], b_mix[0] = self.para_mix(comp[0])

        # find the equilibrium pressure and mol fraction of liquid phase
        comp[1] = self.mix.values
        for P in np.arange(np.min(self.Psat), 50, 0.01):
            delta_K_sum = 1e5
            K_sum_pre = 1
            while delta_K_sum > 0.005:
                ln_phi = np.empty((2, self.num))
                a_mix[1], b_mix[1] = self.para_mix(comp[1])
                beta_mix = b_mix * P / (R * 10) / self.T
                q_mix = a_mix / b_mix / (R * 10) / self.T
                for i in range(2):
                    # cycle for each phase
                    Z_guess = 1e-5 if i == 1 else 0.8
                    Z_mix[i] = fsolve(self.func_z(beta_mix[i], q_mix[i], status=i), [Z_guess])
                    I_mix[i] = np.log((Z_mix[i] + beta_mix[i]) / Z_mix[i])
                    for j in range(self.num):
                        # cycle for each component
                        a_ba[i, j] = (np.sum((self.a[j] * self.a) ** 0.5 * comp[i]) * 2 - a_mix[i])  # * (1 - 0.094)
                        q_ba[i, j] = q_mix[i] * (1 + a_ba[i, j] / a_mix[i] - self.b[j] / b_mix[i])
                        if Z_mix[i] - beta_mix[i] >= 0:
                            ln_phi[i, j] = self.b[j] * (Z_mix[i] - 1) / b_mix[i] - \
                                           np.log(Z_mix[i] - beta_mix[i]) - q_ba[i, j] * I_mix[i]
                        else:
                            ln_phi[i, j] = -1e10
                phi = np.exp(ln_phi)
                K = phi[1] / phi[0]
                K_sum_cal = np.sum(comp[0] / K)
                comp[1] = comp[0] / K / K_sum_cal
                delta_K_sum = abs(K_sum_cal - K_sum_pre)
                K_sum_pre = K_sum_cal
            if abs(K_sum_cal - 1) < 0.01:
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
                return res
            elif abs(K_sum_cal - 1) < abs(K_sum_pre - 1):
                res = {'P': P, "K": K, "phi": phi, "comp": comp}
        return res


# exp = [0.209330615, 0.670652596, 0.028540247, 0.043940264]
# ['CO2', 'H2', 'Methanol', 'H2O'] ['CO2', 'H2', 'Methanol', 'H2O', 'CO'] 'CO2', 'Methanol', 'H2O']
exp = [0.209059763, 0.669913119, 0.029543594, 0.043759505, 0.047724019]
mix = pd.Series(exp, index=['CO2', 'H2', 'Methanol', 'H2O', 'CO'])  # 'Methane', 'Butane' 'N2', 'Methane'
aa = VLE(T=323.15, comp=mix)

print(aa.dew_p)
