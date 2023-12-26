import time

from read import ReadData
from simulator import Simulation
# from sim import Simulation

a = time.time()
model = 'BU'
RED = '\033[91m'
ENDC = '\033[0m'

# prepare data for the simulation
in_data = ReadData(kn_model=model)
reactor_data = in_data.reactor_data
feed_data = in_data.feed_data
chem_data = in_data.chem
insulator_data = in_data.insulator_data

n = 0
for i in range(feed_data.shape[0]):
    for j in range(reactor_data.shape[0]):
        for k in range(insulator_data.shape[0]):
            # insulator_data['Din'].iloc[k] = reactor_data['Dt'].iloc[j]
            # sim = Simulation(reactor_data.iloc[j], chem_data, feed_data.iloc[i],
            #                  insulator_data.iloc[k], eos=1, drop=1)
            # sim.sim(save_profile=1, loop='indirect', rtol=0.0001, r_target=None)
            try:
                sim = Simulation(reactor_data.iloc[j], chem_data, feed_data.iloc[i],
                                 insulator_data.iloc[k], eos=1, drop=0)
                sim.sim(save_profile=1, loop='indirect', rtol=0.0001, r_target=None)
                del sim
            except ValueError as ve:
                # print(f'{ve}')
                print(f'{RED}{ve}{ENDC}')
                pass
            n += 1

b = time.time()
print(b - a)
