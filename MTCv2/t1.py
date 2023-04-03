import numpy as np

# a = np.arange(10).reshape((2,5))
# b = np.array([1,2])
# b = np.repeat(b,5).reshape(2,5)
# print(b)
# print(a)
# c = b*a
# print(c)
# print(np.sum(c, axis=0))
# c = np.vstack((c,np.sum(c,axis=0)))
# print(c)

from CoolProp.CoolProp import PropsSI
#
# a = ['CO2','H2']
# mix = 'HEOS::CO[0.04]&H2[0.82]&CO2[0.03]&N2[0.11]'
# P = 5e6
# print(PropsSI('D', 'T', 500, 'P', P, mix))

a = np.array([1,2,3,4,5,6])
# print(a[[0, 1, ]])
print(np.delete(a, [2, 3]))

print(np.insert(a, 2, [1, 2]))