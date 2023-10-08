import numpy as np
from scipy.optimize import minimize
# import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI


class EquilibriumCalculator:
    def __init__(self, hf_params, sf_params, temperature, pressure):
        self.hf_params = hf_params
        self.sf_params = sf_params
        self.temperature = temperature
        self.pressure = pressure

        # Calculate enthalpy of formation, entropy of formation, and Gibbs energy of formation
        self.dHfs = np.array([self.calculate_dHf(params) for params in hf_params])
        self.dSfs = np.array([self.calculate_dSf(params) for params in sf_params])
        self.dGfs = self.calculate_dGf(self.dHfs, self.dSfs)
        print(self.dHfs)
        print(self.dGfs)

    def calculate_dHf(self, params):
        T_array = np.array([1, self.temperature, self.temperature ** 2])
        return np.sum(T_array * params)

    def calculate_dSf(self, params):
        T_array = np.array([1, 1 / self.temperature, self.temperature])
        return np.sum(T_array * params)

    def calculate_dGf(self, dH, dS):
        return dH - self.temperature * dS / 1000

    def calculate_dHr(self, feed, product):
        return np.sum(self.dHfs * (product - feed))

    def calculate_dGr(self, feed, product):
        return np.sum(self.dGfs * (product - feed))

    def objective_function(self, x):
        dGf = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] <= 1e-10:
                dGf[i] = 0
            else:
                dGf[i] = self.dGfs[i] + 8.314 * self.temperature * np.log(x[i] / np.sum(x) * self.pressure / 1) / 1000
        dGr = np.sum(dGf * x)

        return dGr

    # @staticmethod
    # def equality_constraints(x, feed, element_count):
    #     return [np.sum((feed - x) * element_count[i]) for i in range(3)]

    @staticmethod
    def equality_constraints(x, feed, element_count):
        return [np.sum((x - feed) * element_count[i]) for i in range(3)]

    def minimize_equilibrium(self, feed, element_count):
        # feed = feed / np.sum(feed)  # convert feed to molar fraction
        # Define the initial guess for the molar fractions of the product
        x0 = feed * 0.8
        x0[2:] = feed[0] * 0.2

        # Define bounds for the variables (product fractions should be between 0 and 1)
        bounds = [(0, 1) for _ in range(len(x0))]
        # Combine equality and inequality constraints
        constraint_rule = [{'type': 'eq', 'fun': lambda x: self.equality_constraints(x, feed, element_count)},
                           {'type': 'ineq', 'fun': lambda x: x}]  # ensure >= 0

        # Solve the optimization problem using SciPy's minimize function
        result = minimize(lambda x: self.objective_function(x), x0, constraints=constraint_rule)

        return result.x, result.fun


if __name__ == "__main__":
    # # Parameters for calculating enthalpy of formation
    # # CO2 H2 CH3OH H2O CO
    hf_para = np.array([[-393.880, 0.001910, -2.1060E-06],
                        [0, 0, 0],
                        [-187.990, -0.049757, 2.1603E-05],
                        [-238.410, -0.012256, 2.7656E-06],
                        [-112.560, 0.009255, -7.8431E-06]])

    # Parameters for calculating entropy of formation
    # CO2 H2 CH3OH H2O CO
    sf_para = np.array([[5.5988, -529.35, -0.0027],
                        [0, 0, 0],
                        [-175.95, 13509, 0.0028],
                        [7.6634, 945, -0.0023],
                        [100.86, -2623.6, -0.0084]])
    #
    # T, P = 273+150, 10
    #
    # # Define the feed and product compositions
    feed_comp = np.array([1, 3, 0, 0, 0])  # CO2 H2 CH3OH H2O CO
    product_comp = np.array([0, 0, 1, 1, 0])  # CO2 H2 CH3OH H2O CO

    element = np.array([[1, 0, 1, 0, 1],
                        [0, 2, 4, 2, 0],
                        [2, 0, 1, 1, 1]])

    #
    # # res = [np.sum((product_comp - feed_comp) * element[i]) for i in range(3)]
    # # print(res)
    # Create an instance of EquilibriumCalculator
    equilibrium_calculator = EquilibriumCalculator(hf_para, sf_para, 200+273.15, 30)
    #
    # Calculate equilibrium
    optimal_product_fractions, min_gibbs_energy = equilibrium_calculator.minimize_equilibrium(feed_comp, element)
    print(equilibrium_calculator.objective_function(feed_comp))
    p1 = np.array([0.751991, 2.31926, 0.216366, 0.248009, 0.0316429])
    print(equilibrium_calculator.objective_function(p1))
    print(EquilibriumCalculator.equality_constraints(p1, feed_comp, element))
    # Print the results
    print("Optimal molar fractions of the product:", np.round(optimal_product_fractions, 2))
    print(EquilibriumCalculator.equality_constraints(optimal_product_fractions, feed_comp, element))
    print("Minimum Gibbs free energy:", min_gibbs_energy)
    # P = pd.Series([0,1,2], index=["a","b","c"])
    # print(P[0])
