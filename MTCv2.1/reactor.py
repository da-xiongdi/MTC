import json
import numpy as np
import scipy
from CoolProp.CoolProp import PropsSI
import pandas as pd
from matplotlib import pyplot as plt

R = 8.314  # J/mol/K


class Reaction:
    """
    simulation of adiabatic reactor for conversion of CO2 to CH3OH
    """

    def __init__(self, kn_model='BU'):
        self.kn_model = kn_model
        if kn_model == 'GR':
            chem_path = 'in_chem_GR.json'
        elif kn_model == 'BU':
            chem_path = 'in_chem_BU_revised.json'
        in_path = {'chem': chem_path, 'reactor': 'in_reactor.json', 'feed': 'in_feed.json'}
        in_data = dict()
        for key, values in in_path.items():
            with open(values) as f:
                in_data[key] = json.load(f)

        # reactor parameters
        self.L, self.Dt = in_data['reactor']["reactor"]['L'], in_data['reactor']["reactor"]['Dt']  # length, m
        self.nrt = in_data['reactor']["reactor"]['nt']  # number of the reaction tube
        self.phi = in_data['reactor']["reactor"]["phi"]  # void of fraction
        self.rhoc = in_data['reactor']["reactor"]["rhoc"]  # density of catalyst, kg/m3
        self.insulator_para = in_data['reactor']["insulator"]

        # prescribed chem data of reaction
        self.chem_data = in_data['chem']
        self.comp_list = in_data['chem']["comp_list"]
        self.react_num = len(in_data['chem']["kr"])
        self.react_sto = np.empty((self.react_num, 5))
        # self.react_dH = np.empty(self.react_num)
        for i in range(self.react_num):
            key = str(i + 1)
            self.react_sto[i] = in_data['chem']["stoichiometry"][key]
            # self.react_dH[i] = in_data['chem']["heat_reaction"][key]

        # feed gas parameter
        self.F0 = np.zeros(len(self.comp_list))  # component of feed gas, mol/s; ndarray
        self.P0, self.T0 = in_data['feed']["condition"]["P"], in_data['feed']["condition"]["T"]  # P0 bar, T0 K
        self.v0 = in_data['feed']["condition"]["Sv"] / self.nrt  # volumetric flux per tube, m3/s

        self.Ft0 = self.P0 * 1e5 * self.v0 / R / self.T0  # total flux of feed,mol/s
        if in_data['feed']["condition"]["recycle"] == "off":  # fresh stream
            self.F0[0] = self.Ft0 / (in_data['feed']["condition"]["H2/CO2"] + 1)
            self.F0[1] = self.Ft0 - self.F0[0]
        elif in_data['feed']["condition"]["recycle"] == "on":  # recycled stream
            self.F0 = np.array([float(i) for i in in_data['feed']["feed"].split('\t')])

    @staticmethod
    def react_H(T, in_dict):
        dH = np.zeros(2)
        i = 0
        for key, value in in_dict["heat_reaction"].items():
            dH[i] = -(value[0] * T + value[1])
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
        # react_comp_rate = np.hstack((react_comp_rate, np.array([0, 0, 0, 0]).reshape(4, 1)))

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
        dF_react = self.rate_gr(T, Pi) if self.kn_model == 'GR' else self.rate_bu(T, Pi)

        # calculate the change of enthalpy due to reaction, kJ/(kg_cat s)
        dH_react = self.react_H(T, self.chem_data)
        dH = np.matmul(dF_react[:-1, 0], dH_react.T)

        # calculate the heat capacity of each component, cp*n, J/(s K)
        heat_capacity = 0
        for i in range(5):
            # read the heat capacity for each component, J/(mol K)
            cp = PropsSI('CPMOLAR', 'T', T, 'P', Pi[i] * 1e5, self.comp_list[i]) if Pi[i] > 0 else 0
            heat_capacity += cp * F_dict[i]
        dT = dH * 1e3 / heat_capacity  # K/kg_cat

        return dF_react[-1], dT

    def simulator(self):
        """
        ode for the concentration distribution along the channel
        radial distribution
        the origin is located at the center of the circle
        :param inner_cond: temperature, molar fraction, and radius at inside;list
        :param outer_cond: temperature, molar fraction, and radius at outside; list
        :param P: pressure of mixture
        :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
        :return: concentration and its slop
        """

        # model for ode solver
        P = self.P0

        def model(z, y):
            # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, F_N2, T]
            F_in = np.array(y[:-1])
            temp = self.balance(y[-1], P, F_in)
            dl2dw = np.pi * ((self.Dt ** 2) / 4) * self.rhoc * self.phi
            dF_dz = temp[0] * dl2dw
            dT_dz = temp[1] * dl2dw
            return np.append(dF_dz, [dT_dz])

        z_span = [0, self.L]
        ic = np.append(self.F0, [self.T0])
        # ode_method = 'RK45' if self.kn_model == 'GR' else 'BDF'
        res = scipy.integrate.solve_ivp(model, z_span, ic, method='BDF', t_eval=np.linspace(0, self.L, 100))
        data = res.y
        r = (data[0, -1] - data[0, 0]) / data[0, 0]
        print(r)
        save_data = pd.DataFrame(data.T, columns=self.comp_list + ['T'])
        save_data.to_excel('result/result_adiabatic_BU_%s_%s_%s_%s_%s.xlsx' %
                           (self.Dt, self.L, self.T0, self.P0, self.v0))


class Insulator(Reaction):
    def __init__(self):
        super().__init__()

        # insulator parameters
        self.nit = self.insulator_para["nt"]  # tube number of the insulator
        self.status = self.insulator_para['status']
        self.Din, self.Do = self.insulator_para['Din'], self.insulator_para['Do']
        self.Tc = self.insulator_para['Tc']
        self.location = self.insulator_para["io"]

    @staticmethod
    def cold_comp(Pi_in, P_sat_out):
        """
        calculate the partial pressure of each gas in the cold side
        :param Pi_in: partial pressure of each gas in the hot side, bar;ndarray
        :param P_sat_out: saturated pressure of condensates in the cold side, bar;ndarray
        :return: partial pressure of each gas at Tc, bar; ndarray
        """
        P_t = np.sum(Pi_in)
        xi_in = Pi_in / P_t
        xi_ncon = np.delete(xi_in, [2, 3])
        xi_ncon = xi_ncon / np.sum(xi_ncon)
        Pi_ncon_out = (P_t - np.sum(P_sat_out)) * xi_ncon
        Pi_out = np.insert(Pi_ncon_out, 2, P_sat_out)
        return Pi_out

    @staticmethod
    def p_sat(T, xi, dx=0.01):
        """
        calculate the saturated pressure of condensates for liquid with specific ratio
        :param T: temperature, K
        :param xi: molar ratio of CH3OH in liquid
        :param dx: step size of xi in the calculation
        :return: saturated pressure of condensate, bar; ndarray
        """
        mix_liquid = 'HEOS::Methanol[%s]&H2O[%s]' % (xi[0], xi[1])
        Pl_sat = PropsSI('P', 'T', T, 'Q', 0, mix_liquid)
        diff_x = 1e5
        for xg_H2O in np.arange(0, 1, dx):
            mix_gas = 'HEOS::Methanol[%s]&H2O[%s]' % (1 - xg_H2O, xg_H2O)
            Pv_sat = PropsSI('P', 'T', T, 'Q', 1, mix_gas)
            temp = abs(Pl_sat - Pv_sat)
            if temp < diff_x:
                xg_H2O_sel = xg_H2O
                diff_x = temp
                if diff_x / Pl_sat < 0.05: break
        pi_v = Pv_sat * np.array([1 - xg_H2O_sel, xg_H2O_sel]) * 1e-5
        return pi_v

    @staticmethod
    def mixture_property(T, Pi_gas, component):
        """
        calculate the properties of gas mixture
        :param T: gas temperature, K
        :param Pi_gas: partial pressure, bar; pd.Serize
        :param component: component of gas, list
        :return: thermal conductivity W/(m K), viscosity Pa s, heat capacity J/mol/K; pd.series
        """
        # prepare data for calculation
        n = len(Pi_gas.index)  # number of gas species

        [cp, k, vis, M] = np.empty((4, n))
        mol_fraction = Pi_gas.values / np.sum(Pi_gas.values)  # mol fraction of gases
        Pi_gas = Pi_gas * 1e5  # convert bar to pa

        i = 0
        # calculate the properties of pure gases
        for comp in component:
            gas = "N2" if comp == "CO" else comp  # "CO" is not available in CoolProp
            # thermal conductivity, W/(m K)
            if Pi_gas[comp] < 1000:
                k[i] = vis[i] = cp[i] = 1e-5
            else:
                k[i] = PropsSI('L', 'T', T, 'P', Pi_gas[comp] - 100, gas)
                # viscosity, Pa S
                vis[i] = PropsSI('V', 'T', T, 'P', Pi_gas[comp] - 100, gas)
                # heat capacity, J/(mol K)
                cp[i] = PropsSI('CPMOLAR', 'T', T, 'P', Pi_gas[comp] - 100, gas)
            # molar weight, g/mol
            M[i] = PropsSI('MOLARMASS', 'T', T, 'P', 1e5, gas)
            i += 1

        # calculate the properties of mixture
        cp_m = np.sum(cp * mol_fraction)
        phi, denominator = np.ones((n, n)), np.ones((n, n))  # Wilke coefficient
        vis_m, k_m = 0, 0
        for i in range(n):
            for j in np.arange(n):
                phi[i, j] = (1 + (vis[i] / vis[j]) ** 0.5 * (M[j] / M[i]) ** 0.25) ** 2 / (8 * (1 + M[i] / M[j])) ** 0.5
                denominator[i, j] = mol_fraction[j] * phi[i, j] if i != j else 0
            vis_m += mol_fraction[i] * vis[i] / np.sum(denominator[i])
            k_m += mol_fraction[i] * k[i] / np.sum(denominator[i])
        return pd.Series([k_m, vis_m, cp[2], cp[3], cp_m],
                         index=["k", "vis", 'cp_' + Pi_gas.index[2], 'cp_' + Pi_gas.index[3], "cp_m"])

    @staticmethod
    def ode_single(inner_cond, outer_cond, P, properties):
        """
        ode for the concentration distribution along the channel, only one condensate
        :param inner_cond: temperature, molar fraction, and radius at inside;list
        :param outer_cond: temperature, molar fraction, and radius at outside; list
        :param P: pressure of mixture, bar
        :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
        :return: concentration and its slop
        """

        P = P * 1e5
        [cp, D, k] = properties
        [T1, c1, r1] = inner_cond
        [T2, c2, r2] = outer_cond

        def model(z, y):
            [x, N, T, dTdz] = y
            dx_dz = -N * (1 - x) / D / (P / R / T)
            dN_dz = -N / z
            d2T_dz2 = dTdz / z - N * cp * dTdz / k
            return np.vstack((dx_dz, dN_dz, dTdz, d2T_dz2))

        def bound(ya, yb):
            return np.array([ya[0] - c1, ya[2] - T1, yb[0] - c2, yb[2] - T2])

        xa, xb = r1, r2
        xini = np.linspace(xa, xb, 11)
        yini = np.zeros((4, xini.size))
        yini[0] = np.linspace(c1, c2, xini.size)
        yini[1] = (c1 - c2) / (r1 - r2)
        yini[2] = np.linspace(T1, T2, xini.size)
        yini[3] = (T1 - T2) / (r1 - r2)
        res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-10, max_nodes=1000)
        xsol = np.linspace(xa, xb, 200)
        ysol = res.sol(xsol)
        return ysol

    @staticmethod
    def ode_multi(inner_cond, outer_cond, P, properties, r):
        """
        ode for the concentration distribution along the channel, two condensates
        :param inner_cond: temperature, molar fraction, and radius at inside;list
        :param outer_cond: temperature, molar fraction, and radius at outside; list
        :param P: pressure of mixture, bar
        :param properties: heat capacity, diffusion coefficient, thermal conductivity of mixture; list
        :return: concentration and its slop
        """
        P = P * 1e5  # convert bar to pa
        [cp_c, cp_d, D_c, D_d, k] = properties
        [T1, x_c1, x_d1, r1] = inner_cond
        [T2, x_c2, x_d2, r2] = outer_cond
        D_cd = 2.6e-5

        def model(z, y):
            [xc, xd, Nd, T, dTdz] = y
            Nc = r * Nd
            dxd_dz = (-Nd * ((1 - xc - xd) / D_d + xc / D_cd) + Nc * xd / D_cd) / (P / R / T)
            dxc_dz = (-Nc * ((1 - xc - xd) / D_c + xd / D_cd) + Nd * xc / D_cd) / (P / R / T)
            dNd_dz = -Nd / z
            d2T_dz2 = -dTdz / z + dTdz * cp_c * Nc / k + dTdz * cp_d * Nd / k
            return np.vstack((dxc_dz, dxd_dz, dNd_dz, dTdz, d2T_dz2))

        def bound(ya, yb):
            return np.array([ya[0] - x_c1, ya[1] - x_d1, ya[3] - T1,
                             yb[0] - x_c2, yb[3] - T2])

        xa, xb = r1, r2
        xini = np.linspace(xa, xb, 200)
        yini = np.zeros((5, xini.size))
        yini[0] = np.linspace(x_c1, x_c2, xini.size)
        yini[1] = np.linspace(x_d1, x_d2, xini.size)
        yini[2] = -0.01
        yini[3] = np.linspace(T1, T2, xini.size)
        yini[4] = (T1 - T2) / (r1 - r2)
        res = scipy.integrate.solve_bvp(model, bound, xini, yini, tol=1e-8, max_nodes=1000)
        xsol = np.linspace(xa, xb, 200)
        ysol = res.sol(xsol)
        return ysol

    def flux(self, Th, P, F_dict):
        """
        calculate the diffusional flux
        :param Th: temperature of gas in the reactor, K
        :param P: pressure of gas in the reactor, bar
        :param F_dict: gas component in the reactor, mol/s; ndarray
        :return:
        """
        # calculate the correction to volumetric flow rate (m3/s)
        # calculate the partial pressure
        Ft = np.sum(F_dict)
        v = self.v0 * (self.P0 / P) * (Th / self.T0) * (Ft / self.Ft0)
        Pi = F_dict * R * Th / v * 1e-5  # bar

        # insulator parameter
        radium = [self.Din / 2, self.Do / 2]
        position = 0 if self.location == "in" else 1  # 1 means the reactor in the shell side

        # calculate the partial pressure
        Pi_h = pd.Series(Pi, index=self.comp_list, dtype="float")  # pressure of gases in the reactor, bar
        if Pi_h["Methanol"] < 1e-5:
            # if there is no reacted gas, end the calculation
            return np.zeros(len(self.comp_list)), 0, 0
        xi_h = Pi_h / P

        # to judge if the partial pressure of condensate are large enough
        xi_condensate = xi_h["Methanol"] + xi_h["H2O"]
        mix_condensate = 'HEOS::Methanol[%s]&H2O[%s]' % (xi_h["Methanol"] / xi_condensate, xi_h["H2O"] / xi_condensate)
        Pv_sat = PropsSI('P', 'T', self.Tc, 'Q', 1, mix_condensate) * 1e-5
        if (Pi_h["Methanol"] + Pi_h["H2O"]) < Pv_sat:
            # if the partial pressure of condensate is low, only heat diffusion while no mass diffusion
            mix_pro_ave = self.mixture_property((self.Tc + Th) / 2, Pi_h, self.comp_list)
            property_h = self.mixture_property(Th, Pi_h, self.comp_list)
            k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator
            qcv = -2 * np.pi * k_e * (self.Tc - Th) / np.log(radium[1 - position] / radium[position])
            dT = qcv / Ft / property_h["cp_m"]
            dT = -dT if self.location == 'in' else dT
            return np.zeros(len(self.comp_list)), dT, 0

        # to determine the molar flux of condensate
        # for r_CH3OH_H2O in np.arange(0.1, 1, 0.1):
        r_CH3OH_H2O = 0.45  # guess a molar ratio of liquid
        Pi_c_cond = self.p_sat(self.Tc, [r_CH3OH_H2O / (1 + r_CH3OH_H2O), 1 / (1 + r_CH3OH_H2O)])
        Pi_c = pd.Series(self.cold_comp(Pi_h.values, Pi_c_cond), index=self.comp_list)
        xi_c = Pi_c / P

        # calculate the heat conductivity and the heat capacity
        property_h = self.mixture_property(Th, Pi_h, self.comp_list)
        property_c = self.mixture_property(self.Tc, Pi_c, self.comp_list)
        mix_pro_ave = (property_h + property_c) / 2
        k_e = mix_pro_ave["k"] * vof + ks * (1 - vof)  # effective heat conductivity of the insulator

        dev = 0
        if xi_c["Methanol"] > xi_h["Methanol"] * 0.99:
            # only H2O diffused
            cold_cond = [self.Tc, xi_c["H2O"], radium[1 - position]]
            hot_cond = [Th, xi_h["H2O"], radium[position]]
            cond_list = [hot_cond, cold_cond]
            cal_property = [mix_pro_ave["cp_H2O"], 1.4e-5, k_e]
            res = self.ode_single(cond_list[position], cond_list[position - 1], P, cal_property)
            na_H20, na_CH3OH = res[1][-position] * radium[-position] * 2 * np.pi * vof, 0  # mol/(s m)
            qcv = -k_e * res[3][-position] * radium[-position] * 2 * np.pi
            dT = qcv / Ft / property_h["cp_m"]  # k/m
        elif xi_c["H2O"] > xi_h["H2O"] * 0.99:
            # only CH3OH diffuse
            cold_cond = [self.Tc, xi_c["Methanol"], radium[1 - position]]
            hot_cond = [Th, xi_h["Methanol"], radium[position]]
            cond_list = [hot_cond, cold_cond]
            cal_property = [mix_pro_ave["cp_Methanol"], 4.5e-5, k_e]
            res = self.ode_single(cond_list[position], cond_list[position - 1], P, cal_property)
            na_H20, na_CH3OH = 0, res[1][-position] * radium[-position] * 2 * np.pi * vof  # mol/(s m)
            qcv = -k_e * res[3][-position] * radium[-position] * 2 * np.pi
            dT = qcv / Ft / property_h["cp_m"]  # k/m
        else:
            # guess a ratio between N_CH3OH and N_H2O
            # determine the saturated pressure in cold side
            # perform the calculation of diffusional flux
            # the best ratio is selected by comparing the xi_h["H2O"]
            cold_cond = [self.Tc, xi_c["Methanol"], xi_c["H2O"], radium[1 - position]]
            hot_cond = [Th, xi_h["Methanol"], xi_h["H2O"], radium[position]]
            cond_list = [hot_cond, cold_cond]
            cal_property = [mix_pro_ave["cp_Methanol"], mix_pro_ave["cp_H2O"], 4.5e-5, 1.4e-5, k_e]
            # print(cold_cond, hot_cond, cal_property)
            gap_min, rmin, rmax = 1e5, 0, 5
            for r_n_CH3OH_H20 in np.arange(rmin, rmax, 0.01):
                # diffusional governing equation
                res_guess = self.ode_multi(cond_list[position], cond_list[position - 1], P, cal_property, r_n_CH3OH_H20)
                gap_xd = res_guess[1][-1] - cond_list[position - 1][2]
                if abs(gap_xd) < gap_min:
                    gap_min = abs(gap_xd)
                    r_sel = r_n_CH3OH_H20
                    if gap_min / xi_h["H2O"] < 0.05: break
            # calculate the diffusional flux with optimized N_CH3OH/N_H2O
            res = self.ode_multi(cond_list[position], cond_list[position - 1], P, cal_property, r_sel)
            # /m * m2/s * mol/m3 = mol/s/m2
            na_H20 = res[2][-position] * radium[-position] * 2 * np.pi * vof  # mol/(s m)
            na_CH3OH = na_H20 * r_sel
            # calculate the heat flux
            qcv = -k_e * res[4][-position] * radium[-position] * 2 * np.pi
            dT = qcv / Ft / property_h["cp_m"]  # k/m
            dev = gap_min / xi_h["H2O"]
            if dev > 0.1: print([na_CH3OH, na_H20], r_sel, dev, 'dev too big')

        if na_H20 < 0 or na_CH3OH < 0:
            print('err')
            print(cond_list, cal_property, r_sel)
            print("*" * 10)
        dF = np.zeros_like(F_dict)
        dF[2:4] = [na_CH3OH, na_H20]
        if self.location == 'in':
            dF = -1 * dF
            dT = -1 * dT
        return dF, dT, dev

    def simulator2(self):
        """
        ode for the concentration distribution along the channel
        radial distribution
        the origin is located at the center of the circle
        :return: concentration and its slop
        """

        # model for ode solver
        P = self.P0
        performance = []

        def model(z, y):
            # y= [F_CO2, F_H2, F_CH3OH, F_H2O, F_CO, F_N2, T]
            F_in = np.array(y[:-1])
            T = y[-1]
            # volume fraction of catalyst
            r_v_ins_v_react = self.Do ** 2 * self.nit / self.Dt ** 2 / self.nrt if self.location == 'out' else 0
            delta_react = self.balance(T, P, F_in)
            delta_diff = self.flux(T, P, F_in)
            dl2dw = np.pi * ((self.Dt ** 2) / 4) * self.rhoc * self.phi * (1 - r_v_ins_v_react)

            dF_dz = delta_react[0] * dl2dw * self.nrt + delta_diff[0] * self.nit
            dT_dz = delta_react[1] * dl2dw * self.nrt + delta_diff[1] * self.nit

            # performance of diffusional module
            performance.append(z)
            performance.append(T)
            for i in (delta_react[0] * dl2dw).tolist(): performance.append(i)
            for i in (delta_diff[0] * self.nit).tolist(): performance.append(i)
            performance.append(delta_react[1] * dl2dw)
            performance.append(delta_diff[1] * self.nit)

            return np.append(dF_dz, [dT_dz])

        z_span = [0, self.L]
        ic = np.append(self.F0, [self.T0])
        # ode_method = 'RK45' if self.kn_model == 'GR' else 'BDF'
        res = scipy.integrate.solve_ivp(model, z_span, ic, method='BDF', t_eval=np.linspace(0, self.L, 1000))
        data = res.y

        # reactor performance
        r = (data[0, -1] - data[0, 0]) / data[0, 0]
        dF_react_rwgs = data[4][-1]
        dF_react_ch3oh = (data[0, 0] - data[0][-1]) - dF_react_rwgs
        dF_react_h2o = dF_react_rwgs + dF_react_ch3oh
        dF_diff_ch3oh = dF_react_ch3oh - data[2][-1]
        dF_diff_h2o = dF_react_h2o - data[3][-1]
        sp_ch3oh = dF_diff_ch3oh / dF_react_ch3oh
        sp_h2o = dF_diff_h2o / dF_react_h2o
        r_CH3OH_H2O = dF_diff_ch3oh / dF_diff_h2o
        s_react = dF_react_ch3oh / (data[0, -1] - data[0, 0])
        print(r, sp_ch3oh, sp_h2o, r_CH3OH_H2O, s_react)

        n_col = int(len(performance) / 14)
        save_perfomance = pd.DataFrame(np.array(performance).reshape((n_col, 14)))
        save_perfomance.to_excel('result/result_heat_BU_perfomance_%s_%s_%s_%s_%s_%s.xlsx' %
                                 (self.Dt, self.L, self.T0, self.P0, self.Tc, self.v0))
        save_data = pd.DataFrame(data.T, columns=self.comp_list + ['T'])
        save_data.to_excel('result/result_heat_BU_%s_%s_%s_%s_%s_%s.xlsx' %
                           (self.Dt, self.L, self.T0, self.P0, self.Tc, self.v0))


vof = 0.8
ks = 2  # 0.2 W/m K
reactor = Insulator()
reactor.simulator2()

# a = [343, 0.012360681152337636, 0.0039033729954750422, 0.03]
# b = [524.5764062395502, 0.012489109102363472, 0.030998751112742237, 0.015]
# c = [63.685662602388845, 35.503136000940216, 4.5e-05, 1.4e-05, 0.2930081289718913]
# ysol = reactor.ode_multi(b, a, 50, c, 0.6)
#
# xsol = np.linspace(0.015, 0.03, 200)
# # ysol = res.sol(xsol)
# fig, axe = plt.subplots(2, 2)
# # [xc, xd, Nd, T, dTdz]
# axe[0][0].plot(xsol, ysol[0])
# axe[0][0].plot(xsol, ysol[1])
# axe[0][0].legend(["CH3OH", "H2O"])
# axe[0][1].plot(xsol, ysol[2])
# axe[1][0].plot(xsol, ysol[3])
# axe[1][1].plot(xsol, ysol[4])
# plt.show()
#
# print(ysol[1][0],ysol[1][-1])
# print(ysol[1][-1] - b[2])
# print(ysol[1][0] - a[2])
# c1 = [550.9956251927392, 0.006494309709190364, 0.016514917211696244, 0.035]
# c2 = [353, 0, 0, 0.015]
# c3 = [58.02352864647421, 35.9547344923201, 4.5e-05, 1.4e-05, 0.31750891539283604]
# r = 0.14


# gap = 1e5
# for r in np.arange(0, 2, 0.01):
#     res = ode_multi(c2, c1, 50, c3, r)
#     temp = abs(res[1][-1] - c1[2])
#     if temp < gap:
#         r_sel = r
#         gap = temp
#
# print(r_sel, gap)
# ode_multi(c2, c1, 50, c3, r=1.25)

# T = 93+273
# print(PropsSI('P', 'T', T, 'Q', 1, 'Ammonia'))
# mix_liquid = 'HEOS::H2O[0.9]&Ammonia[0.1]'#'HEOS::NH3[%s]&H2O[%s]' % (0.1, 0.9)
# Pl_sat = PropsSI('P', 'T', T, 'Q', 1, mix_liquid)
# print(Pl_sat)

# for i in np.arange(350,380):
#     a.append(PropsSI('P', 'T', i, 'Q', 1, 'Methanol'))
#     b.append(PropsSI('P', 'T', i, 'Q', 1, 'H2O'))
#
# plt.plot(np.arange(350,380),a)
# plt.plot(np.arange(350,380),b)
# plt.show()
