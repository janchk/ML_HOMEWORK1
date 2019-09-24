import numpy as np
import pandas as pd

from src.Lin_reg import lin_regression
from src.parse_csv import read_csv
from src.SGD import gradient_desc
from src.stat_funcs import RMSE, R2


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


# def normalize(x):
#     x = x.trnspose()

# np.no
# for row in x:

def get_csv_data():
    path = "../Dataset/Training/"
    csv_file = path + "Features_Variant_1.csv"
    df = read_csv(csv_file)
    return df.values

if __name__ == "__main__":
    df_array = get_csv_data()

    # X = [row[0:28, 35:53] for row in df_array]
    X = [np.hstack((row[0:29], row[34:54])) for row in df_array]
    X = normalized(X)
    Y = [row[29] for row in df_array]
    Y = normalized(Y).T
    # Y = np.expand_dims(Y, axis=-1)
    betas = np.random.rand(49)

    # gd = gradient_desc(X,Y, betas)
    gd = gradient_desc(X, Y, betas)
    pass
