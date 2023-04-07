from reactor import Reaction
from insulator import Insulation
from simulator import Simulation

sim = Simulation(kn_model='SL', r_CH3OH_H2O=0.5)
sim.simulator()
