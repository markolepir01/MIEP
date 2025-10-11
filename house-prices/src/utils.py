import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse_log(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def log_transform(y):
    return np.log1p(y)

def inverse_log(y_pred):
    return np.expm1(y_pred)