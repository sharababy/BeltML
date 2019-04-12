import numpy as np 


data0 = np.genfromtxt('tseries_ideal_7_5.csv', delimiter=',')
data1 = np.genfromtxt('tseries_misaligned_7_5.csv', delimiter=',')
data2 = np.genfromtxt('tseries_undertension_7_5.csv', delimiter=',')


# adding target values to predict
#  0 = correct 
#  1 = wrong
data0_wt = np.insert(data0, 56, 0, axis=1)
data1_wt = np.insert(data1, 56, 1, axis=1)
data2_wt = np.insert(data2, 56, 1, axis=1)

dataset = np.concatenate((data0_wt,data1_wt,data2_wt),axis=0)

print(dataset.shape)


np.savetxt("tseries_all_7_5_t.csv",dataset,delimiter=',', fmt='%f',)