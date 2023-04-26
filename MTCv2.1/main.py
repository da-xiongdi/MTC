from read import ReadData
from simulator import Simulation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
# 'BU',
for model in ['SL']:
    # prepare data for the simulation

    in_data = ReadData(kn_model=model)
    reactor_data = in_data.reactor_data
    feed_data = in_data.feed_data
    chem_data = in_data.chem
    insulator_data = in_data.insulator_data

    # guess r_CH3OH_H2O accroding to preliminary sim
    lr = LinearRegression()  # 实例化一个线性回归模型

    path = 'D:/document/04Code/PycharmProjects/MTC/MTCv2.1/result/sim_%s_log.csv' % model
    data = pd.read_csv(path)
    r_CH3OH_H2O = data['N_CH3OH_H2O']

    # select_column = ['T', 'P', 'CO/CO2', 'Sv', 'Tc', 'N_CH3OH_H2O']
    # select_data = data.loc[data['Dt'] == 0.03, select_column]
    # X = select_data.drop(['N_CH3OH_H2O'], axis=1)
    # Y = select_data['N_CH3OH_H2O']
    # Xs = StandardScaler().fit_transform(X)
    #
    # Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)
    # lr.fit(Xtrain, Ytrain)  # 使用训练集训练线性回归模型
    # print(lr.score(Xtest, Ytest))
    n = 0
    for i in range(feed_data.shape[0]):
        for j in range(reactor_data.shape[0]):
            for k in range(insulator_data.shape[0]):
                insulator_data['Din'].iloc[k] = reactor_data['Dt'].iloc[j]

                # x_r_CH3OH_H2O = np.array([feed_data['T'].iloc[i], feed_data['P'].iloc[i],
                #                           feed_data['CO/CO2'].iloc[i], feed_data['Sv'].iloc[i],
                #                           insulator_data['Tc'].iloc[k]])
                y_r_CH3OH_H2O = r_CH3OH_H2O.iloc[n]  # lr.predict(x_r_CH3OH_H2O.reshape(1, -1))[0]
                sim = Simulation(reactor_data.iloc[j], chem_data, feed_data.iloc[i],
                                 insulator_data.iloc[k], r_CH3OH_H2O=y_r_CH3OH_H2O)
                sim.simulator()
                n += 1
