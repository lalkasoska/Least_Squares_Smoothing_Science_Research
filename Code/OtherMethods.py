import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor


def kernel(x, y):
    # Fit the Kernel regression model
    clf = KernelRidge(kernel='rbf', alpha=0.6, gamma=1)
    clf.fit(x, y)
    # Predict on new data
    y_pred = clf.predict(x)
    return y_pred


def local(x, y):
    lowess = sm.nonparametric.lowess(y, x, frac=0.3)
    y_pred = lowess[:, 1]
    return y_pred


def gauss(x, y):
    # y = np.sin(X).ravel()
    # Define the kernel function
    kernel = RBF(length_scale=1.0)
    # Fit the Gaussian process regression model
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)
    gpr.fit(x, y)

    # Predict on new data
    y_pred, y_std = gpr.predict(x, return_std=True)
    return y_pred


def regTrees(x, y):
    # Fit the regression tree model
    reg_tree = DecisionTreeRegressor(max_depth=8)
    reg_tree.fit(x, y)

    # Predict on new data
    y_pred = reg_tree.predict(x)
    return y_pred


def randomForest(x, y):
    # Fit the random forest regression model
    rf = RandomForestRegressor(n_estimators=1000, max_depth=14)
    rf.fit(x, y)
    # Predict on new data
    y_pred = rf.predict(x)
    return y_pred


from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(500, 100, 20, 10), activation='tanh', solver='adam', max_iter=5000, alpha=0,
                   learning_rate_init=0.001)


def MLP(x, y):
    # Fit the MLP regression model

    mlp.fit(x, y)

    # Predict on new data

    y_pred = mlp.predict(x)
    return y_pred


from sklearn.neighbors import KNeighborsRegressor


def kNeighbours(x, y):
    n_neighbors = 8
    knn_reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_reg.fit(x, y)

    # Predict on new data

    y_pred = knn_reg.predict(x)
    return y_pred


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import  pandas as pd
import statsmodels.api as sm
def lowess(x, y):
    x = [elem[0] for elem in x]
    y = np.array(y)
    # Apply loess smoothing
    y_pred = sm.nonparametric.lowess(y, x, frac=0.1, it=0,return_sorted=False)
    return y_pred

from loess.loess_1d import loess_1d
def loes(x, y):
    x = [elem[0] for elem in x]
    x = np.array(x)  # .reshape(-1,1)
    y = np.array(y)
    # Apply loess smoothing
    _, y_pred, _ = loess_1d(x, y, xnew=None, degree=1, frac=0.1,
                                npoints=None, rotate=False, sigy=None)
    return y_pred



from scipy.signal import savgol_filter
def savgol(x,y):
    y_pred=savgol_filter(y, 25, 2)
    return y_pred
# Plot the results
df = pd.read_excel('../dots.xlsx')
x = df['x'].tolist()
y = df['f(x)'].tolist()
y_real = df['f(x) Real'].tolist()
x = np.array(x)#.reshape(-1,1)
y = np.array(y)

# plt.scatter(x, y, color='red',alpha=0.2, label='data')
# plt.plot(x,y_real,'g', label="RealFunc")
# # plt.plot(x, kernel(x,y), color='red', linestyle='-', linewidth=3.0,label='Kernel')
# # #plt.plot(x, local(x,y), color='blue', linestyle='--',linewidth=3.0, label='Local')
# # plt.plot(x, gauss(x,y), color='green',linestyle=':', linewidth=3.0,label='Gauss')
# # plt.plot(x, regTrees(x,y), color='purple',linestyle='-.',linewidth=3.0,label='Regression Tree')
# # plt.plot(x, randomForest(x,y), color='blue', linestyle='--',linewidth=3.0,label='Random Forest')
# #plt.plot(x, MLP(x,y), color='blue', linestyle='--',linewidth=3.0,label='Neural Network')
# plt.plot(x, savgol(x,y),'bo',marker='.', linestyle='',linewidth=3.0,label='SavGol')
# plt.legend()
# plt.show()
