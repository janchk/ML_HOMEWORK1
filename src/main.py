import numpy as np
import pandas as pd

# from src.Lin_reg import lin_regression
from math import sqrt

from src.parse_csv import read_csv
from src.SGD import *


# from src.stat_funcs import RMSE, R2


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_csv_data():
    path = "../Dataset/Training/"
    csv_file = path + "Features_Variant_1.csv"
    df = read_csv(csv_file)
    return df.values


def validate(y, test_features, betas, bias):
    predictions = lin_regression(test_features, betas, bias)

    r2_val = R2(y, predictions)
    rmse_val = sqrt(mse(y, predictions))
    print("R2 is {}", r2_val)
    print("RMSE is {}", rmse_val)
    print("R2/RMSE is {}", r2_val/rmse_val)



if __name__ == "__main__":
    df_array = get_csv_data()

    # X = [row[0:28, 35:53] for row in df_array]
    X = [np.hstack((row[0:29], row[34:54])) for row in df_array]
    X = normalized(X)
    Y = [row[29] for row in df_array]
    # Y = normalized(Y).T
    Y = np.expand_dims(Y, axis=-1)
    # betas = np.ones(49)
    betas = np.expand_dims(np.random.rand(49), axis=-1)

    (final_betas, final_bias) = gradient_desc(X, Y, betas)



    pass
