import numpy as np
from statsmodels.tsa.seasonal import  STL
import matplotlib.pyplot as plt
import pandas as pd


def STL_Process(Panda_object,period = 30):

    stl = STL(Panda_object, period= period, robust=True)
    res_robust = stl.fit()
    res_robust.plot()
    plt.show()

    trend = res_robust.trend.values
    resid = res_robust.resid.values
    seasonal =  res_robust.seasonal.values
    weights = res_robust.weights
    features = res_robust.observed.values

    features = features[:,np.newaxis]
    trend = trend[:,np.newaxis]
    resid = resid[:,np.newaxis]
    seasonal = seasonal[:,np.newaxis]
    weights = weights[:,np.newaxis]

    features = np.concatenate([features,trend],axis=1)
    features = np.concatenate([features,resid],axis=1)
    features = np.concatenate([features,seasonal],axis=1)
    features = np.concatenate([features,weights],axis=1)
    # print("5k shape:", features.shape)
    return features

if __name__ == "__main__":
    DF = pd.read_csv('../data/code-red.csv')
    Window_DF = DF["imp_wd"][0:827]
    # Window_DF.index = [i for i in range(Window_DF.shape[0])]
    print(Window_DF)
    STL_Process(Window_DF)

