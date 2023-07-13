import numpy as np
from CoolProp.CoolProp import PropsSI
import pandas as pd

R = 8.314  # J/mol/K
ks, vof = 0.2, 0.8 # 1.5 for 0.42 1 for 0.3 0.2 for 0.15


class Reaction:
    """
    basic simulation of CO2 to CH3OH
    energy and mass balance are calculated
    """

    def __init__(self, reactor_para, chem_para, feed_para):

        # reactor parameters
        self.react_para = reactor_para
        self.L1, self.Dt = self.react_para['L1'], self.react_para['Dt']  # length, m
        self.stage = self.react_para["stage"]
        self.L2 = self.react_para["L2"]
        self.nrt = self.react_para['nrt']  # number of the reaction tube
        self.phi = self.react_para["phi"]  # void of fraction
        self.rhoc = self.react_para["rhoc"]  # density of catalyst, kg/m3
        self.recycle = self.react_para['recycle']  # reactor with recycle or not
        self.Uc = self.react_para['Uc']  # total heat transfer coefficient of reactor, W/m2 K, 0 means adiabatic

        # prescribed chem data of reaction
        self.comp_list = ["CO2", "H2", "Methanol", "H2O", "CO"]
        self.chem_data = chem_para
        self.react_num = len(self.chem_data["kr"])
        self.react_sto = np.empty((self.react_num, 5))
        self.kn_model = self.chem_data['kn_model']
        for i in range(self.react_num):
            key = str(i + 1)
            self.react_sto[i] = self.chem_data["stoichiometry"][key]

        # feed gas parameter
        self.feed_para = feed_para
        self.P0, self.T0 = self.feed_para["P"], self.feed_para["T"]  # P0 bar, T0 K
        self.T_feed = self.feed_para["T_feed"]

        if self.feed_para["fresh"] == 1:  # the feed to the plant is fresh stream
            self.F0 = np.zeros(len(self.comp_list))  # component of feed gas, mol/s; ndarray
            # volumetric flux per tube from space velocity
            if self.feed_para["H2"] == 0:
                self.sv = self.feed_para["Sv"]
                # volumetric flux per tube under input temperature and pressure, m3/s
                self.v0 = self.sv * self.L1 * np.pi * self.Dt ** 2 / 4 / 3600 / self.nrt
                self.Ft0 = self.P0 * 1e5 * self.v0 / R / self.T0  # total flux of feed,mol/s
                self.F0[0] = 1 / (1 + 1 * self.feed_para["H2/CO2"] + self.feed_para['CO/CO2']) * self.Ft0
                self.F0[4] = self.F0[0] * self.feed_para['CO/CO2']
                self.F0[1] = self.Ft0 - self.F0[0] - self.F0[4]
                self.H2 = self.F0[1] * 8.314 * 273.15 / 1e5 * 3600  # Nm3/h
            else:
                self.H2 = self.feed_para["H2"]
                self.F0[1] = self.H2 / 3600 * 1e5 / R / 273.15  # mol/s
                self.F0[0] = self.F0[1] / self.feed_para["H2/CO2"]
                self.F0[4] = self.F0[0] * self.feed_para['CO/CO2']
                self.Ft0 = np.sum(self.F0)
                self.v0 = self.Ft0 * R * self.T0 / (self.P0 * 1e5)
                self.sv = self.v0 * self.nrt * 3600 * 4 / self.L1 / np.pi / self.Dt ** 2
            # print(self.sv, self.H2)
        else:  # recycled stream
            self.F0 = self.feed_para[self.comp_list].to_numpy()
            self.Ft0 = np.sum(self.F0)
            self.v0 = self.Ft0 * R * self.T0 / (self.P0 * 1e5)
            self.sv = self.v0 * self.nrt * 3600 * 4 / self.L1 / np.pi / self.Dt ** 2
            self.H2 = self.F0[1] * R * 273.15/1E5

    @staticmethod
    def react_H(T, in_dict):
        dH = np.zeros(len(in_dict["heat_reaction"].keys()))
        i = 0
        for key, value in in_dict["heat_reaction"].items():
            dH[i] = -(value[0] * T + value[1]) * 1e-6
            i += 1
        return dH

    @staticmethod
    def kad(T, in_dict):
        """
        calculate the equilibrium constant of adsorption
        :param T: operating temperature
        :param in_dict: prescribed chemical parameter
        :return: equilibrium constant of adsorption, 1/bar
        """
        adsorption_eq_constant = dict()
        for key, value in in_dict["kad"].items():
            adsorption_eq_constant[key] = value[0] * np.exp(value[1] / T / R)
        return adsorption_eq_constant

    @staticmethod
    def keq(T, in_dict):
        """
        calculate the equilibrium constant
        :param T: operating temperature
        :param in_dict: prescribed chemical parameter
        :return: equilibrium constant
        """
        react_eq_constant = dict()
        for key, value in in_dict["keq"].items():
            react_eq_constant[key] = 10 ** (value[0] / T + value[1])
        return react_eq_constant

    @staticmethod
    def kr(T, in_dict):
        """
        calculate the reaction rate constant
        :param T: operating temperature, K
        :param in_dict: prescribed chemical parameter
        :return: the reaction rate constant, mol kg−1 s−1 bar-1/2
        """
        react_rate_constant = dict()
        for key, value in in_dict["kr"].items():
            react_rate_constant[key] = value[0] * np.exp(value[1] / T / R)
        return react_rate_constant

    @staticmethod
    def mixture_property(T, Pi_gas):
        """
        calculate the properties of gas mixture
        :param T: gas temperature, K
        :param Pi_gas: partial pressure, bar; pd.Serize
        :return: thermal conductivity W/(m K), viscosity Pa s, heat capacity J/mol/K; pd.series
        """
        # prepare data for calculation
        n = len(Pi_gas.index)  # number of gas species
        Pt = np.sum(Pi_gas.values)

        [cp, k, vis, M, rho] = np.ones((5, n)) * 1e-5
        mol_fraction = Pi_gas.values / Pt  # mol fraction of gases
        Pi_gas = Pi_gas * 1e5  # convert bar to pa
        Ti_sat = pd.Series(np.ones(n) * 100, index=Pi_gas.index)
        if 'Methanol' in Pi_gas.index:
            try:
                Ti_sat['Methanol'] = PropsSI('T', 'P', Pi_gas['Methanol'], 'Q', 1, 'Methanol')
            except ValueError:
                Ti_sat['Methanol'] = 300
        if "H2O" in Pi_gas.index:
            try:
                Ti_sat['H2O'] = PropsSI('T', 'P', Pi_gas['H2O'], 'Q', 1, 'H2O')
            except ValueError:
                Ti_sat['H2O'] = 300
        i = 0
        # calculate the properties of pure gases
        for comp in Pi_gas.index:
            gas = "N2" if comp == "CO" else comp  # "CO" is not available in CoolProp
            if Pi_gas[comp] > 1000:
                if T > Ti_sat[comp]:
                    # thermal conductivity, W/(m K)
                    k[i] = PropsSI('L', 'T', T, 'P', Pt, gas)
                    # viscosity, Pa S
                    vis[i] = PropsSI('V', 'T', T, 'P', Pt, gas)
                    # heat capacity, J/(mol K)
                    cp[i] = PropsSI('CPMOLAR', 'T', T, 'P', Pt, gas)
                    # density, kg/m3
                    rho[i] = PropsSI('D', 'T', T, 'P', Pi_gas[comp], gas)
                else:
                    cp[i] = PropsSI('CPMOLAR', 'T', T, 'Q', 1, gas)
                    k[i] = PropsSI('L', 'T', T, 'Q', 1, gas)
                    vis[i] = PropsSI('V', 'T', T, 'Q', 1, gas)
                    rho[i] = PropsSI('D', 'T', T, 'Q', 1, gas)
            else:
                # thermal conductivity, W/(m K)
                k[i] = 0
                # viscosity, Pa S
                vis[i] = 1e-10
                # heat capacity, J/(mol K)
                cp[i] = 0
                # density, kg/m3
                rho[i] = 0
            # molar weight, g/mol
            M[i] = PropsSI('MOLARMASS', 'T', T, 'P', 1e5, gas)
            i += 1

        # calculate the properties of mixture
        cp_m = np.sum(cp * mol_fraction)
        rho_m = np.sum(rho)
        phi, denominator = np.ones((n, n)), np.ones((n, n))  # Wilke coefficient
        vis_m, k_m = 0, 0
        for i in range(n):
            for j in np.arange(n):
                phi[i, j] = (1 + (vis[i] / vis[j]) ** 0.5 * (M[j] / M[i]) ** 0.25) ** 2 / (8 * (1 + M[i] / M[j])) ** 0.5
                denominator[i, j] = mol_fraction[j] * phi[i, j]  # if i != j else 0
            vis_m += mol_fraction[i] * vis[i] / np.sum(denominator[i])
            k_m += mol_fraction[i] * k[i] / np.sum(denominator[i])
        return pd.Series([k_m, vis_m, rho_m, cp[2], cp[3], cp_m],
                         index=["k", "vis", 'rho', 'cp_' + Pi_gas.index[2], 'cp_' + Pi_gas.index[3], "cp_m"])

    def convection(self, T, P, F_dict):
        """
        :param T: temperature of reactor gas, K
        :param P: pressure of reactor, bar
        :param F_dict: molar flow rate of each component, mol/s; ndarray
        :return: convection heat transfer coefficient, W/m2 K
        """
        Ft = np.sum(F_dict)
        Pi = F_dict / Ft * P
        mix_property = self.mixture_property(T, pd.Series(Pi, self.comp_list))
        M = 0.25 * 44 + 0.75 * 2
        Pr = mix_property['vis'] * (mix_property['cp_m'] / (M / 1000)) / mix_property['k']
        v = self.v0 * (self.P0 / P) * (T / self.T0) * (Ft / self.Ft0)  # m3/s
        u = v / (np.pi * self.Dt ** 2 / 4)
        Re = u * self.Dt * mix_property['rho'] / mix_property['vis']
        if Re > 1e4:
            Nu = 0.0265 * Re ** 0.8 * Pr ** 0.3
        elif 2300 < Re < 1e4:
            f = (0.79 * np.log(Re) - 1.64) ** -2
            Nu = f / 8 * (Re - 1000) * Pr / (1 + 12.7 * (f / 8) ** 0.5 * (Pr ** (2 / 3) - 1))
        elif Re < 2300:
            Nu = 3.66
        h = Nu * mix_property['k'] / self.Dt  # W/m K
        return h

    def rate_bu(self, T, Pi):
        """
        calculate the reaction rate
        :param T: operating temperature, K
        :param Pi: partial pressure of each component, bar
        :return: reaction rate of each component for each and all reaction; mol/s/kg_cat
        """
        # convert the partial pressure from ndarray to pd.Series
        Pi = pd.Series(Pi, index=self.comp_list)

        # calculate the reaction constant
        rate_const = self.kr(T, self.chem_data)
        ad_const = self.kad(T, self.chem_data)
        eq_const = self.keq(T, self.chem_data)

        # calculate the rate of each reaction
        react_rate = np.zeros(self.react_num)
        driving = rate_const['1'] * Pi['CO2'] * Pi['H2'] * (
                1 - Pi['H2O'] * Pi["Methanol"] / Pi["H2"] ** 3 / Pi['CO2'] / eq_const['1'])
        inhibiting = (1 + ad_const["H2O/H2"] * Pi['H2O'] / Pi['H2'] +
                      ad_const["H2"] * Pi["H2"] ** 0.5 + ad_const["H2O"] * Pi["H2O"])
        react_rate[0] = driving / inhibiting ** 3

        driving = rate_const['2'] * Pi['CO2'] * (1 - Pi['H2O'] * Pi["CO"] / Pi["H2"] / Pi['CO2'] / eq_const['2'])
        react_rate[1] = driving / inhibiting

        # compute the reaction rate for each component in every reaction
        react_comp_rate = self.react_sto * np.repeat(react_rate, 5).reshape(self.react_num, 5)
        react_comp_rate = np.vstack((react_comp_rate, np.sum(react_comp_rate, axis=0).T))
        # react_comp_rate = np.hstack((react_comp_rate, np.array([0, 0, 0]).reshape(3, 1)))

        return react_comp_rate

    def rate_sl(self, T, Pi):
        """
        calculate the reaction rate
        :param T: operating temperature, K
        :param Pi: partial pressure of each component, bar
        :return: reaction rate of each component for each and all reaction; mol/s/kg_cat
        """
        # convert the partial pressure from ndarray to pd.Series
        Pi = pd.Series(Pi, index=self.comp_list)

        # calculate the reaction constant
        rate_const = self.kr(T, self.chem_data)
        # print(rate_const)
        ad_const = self.kad(T, self.chem_data)
        eq_const = self.keq(T, self.chem_data)

        # calculate the rate of each reaction
        react_rate = np.zeros(self.react_num)
        driving = rate_const['1'] * Pi['CO2'] * Pi['H2'] ** 2 * (
                1 - Pi['H2O'] * Pi["Methanol"] / Pi["H2"] ** 3 / Pi['CO2'] / eq_const['1'])
        inhibiting = (ad_const["H2"] * Pi['H2'] ** 0.5 +
                      ad_const["H2O"] * Pi["H2O"] + Pi["Methanol"])
        react_rate[0] = driving / inhibiting ** 2

        driving = rate_const['2'] * Pi['CO2'] * (1 - Pi['H2O'] * Pi["CO"] / Pi["H2"] / Pi['CO2'] / eq_const['2'])
        react_rate[1] = driving / inhibiting

        # compute the reaction rate for each component in every reaction
        react_comp_rate = self.react_sto * np.repeat(react_rate, 5).reshape(self.react_num, 5)
        react_comp_rate = np.vstack((react_comp_rate, np.sum(react_comp_rate, axis=0).T))
        # react_comp_rate = np.hstack((react_comp_rate, np.array([0, 0, 0]).reshape(3, 1)))

        return react_comp_rate

    def rate_gr(self, T, Pi):
        """
        calculate the reaction rate
        :param T: operating temperature, K
        :param Pi: partial pressure of each component, bar
        :return: reaction rate of each component for each and all reaction; mol/s/kg_cat
        """

        # convert the partial pressure from ndarray to pd.Series
        Pi = pd.Series(Pi, index=self.comp_list)

        # calculate the reaction constant
        rate_const = self.kr(T, self.chem_data)
        ad_const = self.kad(T, self.chem_data)
        eq_const = self.keq(T, self.chem_data)

        # calculate the rate of each reaction
        react_rate = np.zeros(self.react_num)
        driving = rate_const['1'] * ad_const['CO2'] * (
                Pi['CO2'] * Pi['H2'] ** 1.5 - Pi['H2O'] * Pi["Methanol"] / Pi["H2"] ** 1.5 / eq_const['1'])
        inhibiting = (1 + ad_const["CO"] * Pi['CO'] + ad_const["CO2"] * Pi['CO2']) * \
                     (Pi["H2"] ** 0.5 + ad_const["H2O/H2"] * Pi["H2O"])
        react_rate[0] = driving / inhibiting

        driving = rate_const['2'] * ad_const['CO2'] * (Pi['CO2'] * Pi["H2"] - Pi['H2O'] * Pi["CO"] / eq_const['2'])
        react_rate[1] = driving / inhibiting

        driving = rate_const['3'] * ad_const['CO'] * (
                Pi['CO'] * Pi["H2"] ** 1.5 - Pi['Methanol'] / Pi["H2"] ** 0.5 / eq_const['3'])
        react_rate[2] = driving / inhibiting

        # compute the reaction rate for each component in every reaction
        react_comp_rate = self.react_sto * np.repeat(react_rate, 5).reshape(self.react_num, 5)
        react_comp_rate = np.vstack((react_comp_rate, np.sum(react_comp_rate, axis=0).T))

        return react_comp_rate

    def balance(self, T, P, F_dict):
        """
        energy and material balance in the reactor
        :param T: operating temperature, K
        :param P: operating pressure, bar
        :param F_dict: molar flow rate of each component, mol/s; ndarray
        :return: temperature and molar flux variation of gas
        """
        Ft = np.sum(F_dict)  # total molar flow rate

        # calculate the partial pressure
        # calculate the correction to volumetric flow rate (m3/s)
        v = self.v0 * (self.P0 / P) * (T / self.T0) * (Ft / self.Ft0)
        Pi = F_dict * R * T / v * 1e-5  # bar

        # calculate the change of the molar flow rate due to reactions, mol/s/kg_cat
        if self.kn_model == 'GR':
            dF_react = self.rate_gr(T, Pi)
        elif self.kn_model == 'BU':
            dF_react = self.rate_bu(T, Pi)
        elif self.kn_model == 'SL':
            dF_react = self.rate_sl(T, Pi)

        # calculate the change of enthalpy due to reaction, kJ/(kg_cat s)
        dH_react = self.react_H(T, self.chem_data)
        if self.react_num == 3: dH_react[2] = dH_react[0] - dH_react[1]
        dH = np.matmul(dF_react[:-1, 0], dH_react.T)

        # calculate the heat capacity of each component, cp*n, J/(s K)
        heat_capacity = 0
        for i in range(5):
            # read the heat capacity for each component, J/(mol K)
            # print(T, Pi)
            cp = PropsSI('CPMOLAR', 'T', T, 'P', Pi[i] * 1e5, self.comp_list[i]) if Pi[i] > 0 else 0
            heat_capacity += cp * F_dict[i]
        dT = dH * 1e3 / heat_capacity  # K/kg_cat
        res = {
            'mflux': dF_react[-1],
            'tc': heat_capacity,
            'hflux': dH * 1e3,
            'Tvar': dT
        }
        return res

    # F = np.array([0.062228879, 0.198288202, 0.009296752, 0.012506192, 0.012891983])
# comp = ['CO2', 'H2', 'Methanol', 'H2O', 'CO']
# T = 529
# P = 70
#
# from read import ReadData
#
# # prepare data for the simulation
# in_data = ReadData(kn_model='BU')
# reactor_data = in_data.reactor_data
# feed_data = in_data.feed_data
# chem_data = in_data.chem
# insulator_data = in_data.insulator_data
#
# for i in range(feed_data.shape[0]):
#     for j in range(reactor_data.shape[0]):
#         for k in range(insulator_data.shape[0]):
#             insulator_data['Din'].iloc[k] = reactor_data['Dt'].iloc[j]
#
#             a = Reaction(reactor_data.iloc[j], chem_data, feed_data.iloc[i])
#             a.convection(T, P, F, comp)
