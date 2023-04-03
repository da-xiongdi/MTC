import sys

import numpy as np
import scipy.integrate
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
from sympy import *

ks = 0.2  # W/m K
vof = 0.99
do, dc, dm = 5, 3, 3.04  # m
thick = (dm - dc) / 2


class Test:

    def __init__(self, T1, T2, pt, fluid, qmt, x):
        self.T2 = T2
        self.T1 = T1
        self.pt = pt
        self.mass = np.array([PropsSI('M', 'T', T1, 'P', pt, fluid[i]) for i in range(len(fluid))])  # kg/mol
        self.p_rs = pt * np.array([x / 4, x / 4, (1 - x) / 4, 3 * (1 - x) / 4])
        self.rho_rs = np.array([PropsSI('D', 'T', T1, 'P', self.p_rs[i], fluid[i]) for i in range(len(fluid))])
        self.q_mass_rs = qmt * self.rho_rs / np.sum(self.rho_rs)
        self.q_mol_rs = self.q_mass_rs / self.mass
        self.fluid = fluid
        # x_mol = q_mol_rs / sum(q_mol_rs)

    @staticmethod
    def ode(T1, T2, x1, x2, dl):
        """
        ode for the concentration distribution along the channel
        the origin is located at the outside
        :param T1: inside temperature
        :param T2: outside temperature
        :param x1: inside concentration
        :param x2: outside concentration
        :param dl: length of the channel
        :return: slop of the concentration at the inside
        """

        # function
        def dydx(x, y):
            dT = T1 - T2
            dy0 = y[1]
            dy1 = dT * y[1] / dl / (T2 + dT * x / dl)
            return np.vstack((dy0, dy1))

        def bound(ya, yb):
            return np.array([ya[0] - x2, yb[0] - x1])

        xa, xb = 0, dl
        xini = np.linspace(xa, xb, 11)
        # print(xini)
        yini = np.zeros((2, xini.size))
        res = scipy.integrate.solve_bvp(dydx, bound, xini, yini)
        xsol = np.linspace(xa, xb, 100)
        ysol = res.sol(xsol)[1]
        plt.plot(xsol, ysol)
        plt.show()
        # plt.plot(xsol, res.sol(xsol)[0])
        # plt.show()

        return ysol[-1]

    def vapor(self, i, j):
        """
        calculate the heat flux determined by the vapor diffusion
        :param i: condensable gas
        :param j: non-condensable gas, list
        :return:
        """
        Tm = (self.T1 + self.T2) / 2
        p_c_out = PropsSI('P', 'T', self.T2, 'Q', 1, self.fluid[i])
        p_nc_in, k_nc = 0, 0
        for m in j:
            p_nc_in += self.p_rs[m]
            k_nc += self.p_rs[m] * PropsSI('L', 'T', Tm, 'P', self.p_rs[m], self.fluid[m])
        k_nc = k_nc / p_nc_in
        k_c_in = PropsSI('L', 'T', self.T1, 'P', self.p_rs[i], self.fluid[i])
        k_c_out = PropsSI('L', 'T', self.T2, 'Q', 1, self.fluid[i])
        r = PropsSI('Hmass', 'T', self.T2, 'Q', 1, self.fluid[i]) - PropsSI('Hmass', 'T', self.T2, 'Q', 0,
                                                                            self.fluid[i])

        xc_mol_in = self.p_rs[i] / (self.p_rs[i] + p_nc_in)
        xc_mol_out = p_c_out / (self.p_rs[i] + p_nc_in)

        c_in = (self.p_rs[i] + p_nc_in) / 8.314 / self.T1

        k_v = (k_c_in * xc_mol_in + k_nc * (2 - xc_mol_in - xc_mol_out) + k_c_out * xc_mol_out) / 2
        k_e = k_v * vof + ks * (1 - vof)

        na_mass = -1 * 0.1e-4 * c_in * self.ode(self.T1, self.T2, xc_mol_in, xc_mol_out, thick) / (1 - xc_mol_in) * \
                  self.mass[i]  # kg/m2 s
        qcd = na_mass * r * vof
        qcv = (self.T2 - self.T1) * k_e / thick
        qt = qcd + qcv

        return qt, qcd, qcv, na_mass

    def test(self, i, j):
        """
        calculate the heat flux determined by the vapor diffusion
        :param i: condensable gas
        :param j: non-condensable gas, list
        :return:
        """
        Tm = (self.T1 + self.T2) / 2
        p_c_out = PropsSI('P', 'T', self.T2, 'Q', 1, self.fluid[i])
        p_nc_in, k_nc = 0, 0
        for m in j:
            p_nc_in += self.p_rs[m]
            k_nc += self.p_rs[m] * PropsSI('L', 'T', Tm, 'P', self.p_rs[m], self.fluid[m])
        k_nc = k_nc / p_nc_in
        k_c_in = PropsSI('L', 'T', self.T1, 'P', self.p_rs[i], self.fluid[i])
        k_c_out = PropsSI('L', 'T', self.T2, 'Q', 1, self.fluid[i])
        r = PropsSI('Hmass', 'T', self.T2, 'Q', 1, self.fluid[i]) - PropsSI('Hmass', 'T', self.T2, 'Q', 0,
                                                                            self.fluid[i])
        h_c_in = PropsSI('Hmass', 'T', self.T1, 'P', self.p_rs[i], self.fluid[i])
        h_c_out = PropsSI('Hmass', 'T', self.T2, 'Q', 0, self.fluid[i])
        xc_mol_in = self.p_rs[i] / (self.p_rs[i] + p_nc_in)
        xc_mol_out = p_c_out / (self.p_rs[i] + p_nc_in)
        c_in = (self.p_rs[i] + p_nc_in) / 8.314 / self.T1

        k_v = (k_c_in * xc_mol_in + k_nc * (2 - xc_mol_in - xc_mol_out) + k_c_out * xc_mol_out) / 2
        k_e = k_v * vof + ks * (1 - vof)
        x1 = np.linspace(xc_mol_out, xc_mol_in, 10)
        d1 = np.linspace(1e-4, thick - 1e-4, 500)

        d = 0.002
        x = 1 - np.exp(k_e / vof / 3000 / c_in / 0.1e-4 * d / (thick - d)) * (1 - xc_mol_in)
        print(self.ode(self.T1, self.T1, xc_mol_in, x, d) / (1 - xc_mol_in))
        print(c_in * np.log((1 - x) / (1 - xc_mol_in)))

        sys.exit(1)


        gap = 1e10
        for d in d1:
            # print(x)
            # for d in d1:
            # na_mass_1 = -1 * 0.1e-4 * c_in * self.ode(self.T1, self.T1, xc_mol_in, x, d) / (1 - xc_mol_in) * \
            #             self.mass[i]  # kg/m2 s
            x = 1 - np.exp(k_e / vof / 3000 / c_in / 0.1e-4 * d / (thick - d)) * (1 - xc_mol_in)
            if x < 0:
                break
            # print(x)
            na_mass_1 = -1 * 0.1e-4 * c_in * np.log((1 - x) / (1 - xc_mol_in)) * self.mass[i] / d  # kg/m2 s
            # print(na_mass_1)
            na_mass_2 = -1 * 0.1e-4 * c_in * self.ode(self.T1, self.T2, x, xc_mol_out, thick - d, ) / (1 - x) * \
                        self.mass[i]  # kg/m2 s

            qcd = na_mass_2 * r * vof
            qcv = (self.T2 - self.T1) * k_e / (thick - d)
            qt = qcd + qcv
            na_mass_3 = qt / (r + 3000 * (self.T1 - self.T2)) / vof  # qt / (h_c_in - h_c_out) / vof
            # print(na_mass_3)
            # print(na_mass_2)
            # na_mass_3 = -self.q_mass_rs[i] * (r + 3000 * (523 - self.T2)) / qt / np.pi / dc / vof
            temp = abs(na_mass_2 - na_mass_3)  # + abs(na_mass_1 - na_mass_3)  # +
            if temp < gap:
                gap = temp
                lenth1, x_c_mol_1 = d, x
                na_mass = [na_mass_1, na_mass_2, na_mass_3]
        print(gap, lenth1, x_c_mol_1, na_mass)

        # return qt, qcd, qcv, na_mass_2


Ts = np.arange(313 + 1, 383, 1)
n = Ts.shape
q2, q2_cd, q2_cv, n2_mass = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
fluids = ['water', 'methanol', 'CO2', 'hydrogen']
q_mass_in = 2.25
x_conversion = 0.5
tes = Test(523, 314, 6e6, fluids, q_mass_in, x_conversion)
tes.test(0, [2, 3])
# print(np.linspace(0.1, 0.2, 10))
# k = 0
# for T in Ts:
#     tes = Test(523, T, 6e6, fluids, q_mass_in, x_conversion)
#     q2[k], q2_cd[k], q2_cv[k], n2_mass[k] = tes.vapor(0, [2, 3])
#     k += 1
#
# plt.plot(Ts, q2_cd, '*')
# plt.plot(Ts, q2_cv, '+')
# plt.plot(Ts, q2, '-')
# plt.show()
