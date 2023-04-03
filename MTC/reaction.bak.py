import sys

import numpy as np
import scipy.integrate
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
from sympy import *

"""
rec1:CO+2H2=CH3OH
rec2:CO2+H2=CO+H2O
rec3:CO2+3H2=CH3OH+H2O
"""

para_ad = [[7.05e-7, 61700], [6.37e-9, 84000], [2.16e-5, 46800]]  # 1/bar CO2; H2O/H2; CO
para_eq = [[5139, 12.621], [3066, 10.592], [-2073, -2.029]]
para_rate = [[4.89e7, -63000], [9.64e11, -152900], [1.09e5, -87500]]  # mol/s/kg/bar**0.5


def kad(i, T):
    return para_ad[i][0] * np.exp(para_ad[i][1] / T / 8.314)


def keq(i, T):
    return 10 ** (para_eq[i][0] / (T - para_eq[i][1]))


def kr(i, T):
    return para_rate[i][0] * np.exp(para_rate[i][1] / T / 8.314)


def react1(p1, p2, p3, p4, p5, T):
    """

    :param p1: partial pressure of CO2
    :param p2: partial pressure of H2
    :param p3: partial pressure of H2O
    :param p4: partial pressure of CH3OH
    :param p5: partial pressure of CO
    :param T: temperature
    :return:
    """
    rate = kr(0, T) * kad(2, T) * (p5 * p1 ** 1.5 - p4 / p2 ** 0.5 / keq(0, T)) / \
           (1 + kad(2, T) * p5 + kad(0, T) * p1) / (p2 ** 0.5 + kad(1, T) * p3)
    return rate


def react2(p1, p2, p3, p4, p5, T):
    """

    :param p1: partial pressure of CO2
    :param p2: partial pressure of H2
    :param p3: partial pressure of H2O
    :param p4: partial pressure of CH3OH
    :param p5: partial pressure of CO
    :param T: temperature
    :return:
    """
    rate = kr(1, T) * kad(0, T) * (p1 * p2 - p3 * p5 / keq(1, T)) / \
           (1 + kad(2, T) * p5 + kad(0, T) * p3) / (p2 ** 0.5 + kad(1, T) * p3)
    return rate


def react3(p1, p2, p3, p4, p5, T):
    """

    :param p1: partial pressure of CO2
    :param p2: partial pressure of H2
    :param p3: partial pressure of H2O
    :param p4: partial pressure of CH3OH
    :param p5: partial pressure of CO
    :param T: temperature
    :return:
    """
    rate = kr(2, T) * kad(0, T) * (p1 * p2 ** 1.5 - p4 * p3 / p2 ** 0.5 / keq(2, T)) / \
           (1 + kad(2, T) * p5 + kad(0, T) * p1) / (p2 ** 0.5 + kad(1, T) * p3)
    return rate


p1 = 0.6e6
p2 = 1.8e6
p3 = 0.15e6
p4 = 0.15e6
p5 = 0

print(react1(p1, p2, p3, p4, p5, 523))
print(react2(p1, p2, p3, p4, p5, 523))
print(react3(p1, p2, p3, p4, p5, 523))
