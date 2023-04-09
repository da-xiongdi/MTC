from reactor import Reaction
from insulator import Insulation
from simulator import Simulation

re = Reaction(kn_model='BU')
ins = Insulation(re, r_CH3OH_H2O=0.5)
print(ins.kn_model)
sim = Simulation(kn_model='BU', r_CH3OH_H2O=0.5)
# sim.simulator()


