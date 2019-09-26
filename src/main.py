import numpy as np
import os
import random as rnd
import pandas as pd

# from math import sqrt
#
# #Bokeh
# from bokeh.io import curdoc
# from bokeh.layouts import column
# from bokeh.models import ColumnarDataSource
# from bokeh.plotting import figure
#
# from functools import partial
# from threading import Thread
# from tornado import gen

from src.models import LinearRegressionWithGd
# from src.SGD import *

path = "../Dataset/Training/"


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_csv_data(fpath):
    df = pd.read_csv(fpath)
    return df.values


def cv_loo(ds, fnum, num):
    folds = np.array_split(ds, fnum)
    test = folds.pop(num)
    train = np.vstack(folds)
    return train, test


def folding(path):
    fs_iter = [path + fpath for fpath in os.listdir(path) if fpath.endswith(".csv")]
    train = get_csv_data(fs_iter[rnd.randint(0, 4)])
    test = get_csv_data(fs_iter[rnd.randint(0, 4)])

    return train, test


if __name__ == "__main__":

    lrg = LinearRegressionWithGd()

    df_1, df_2 = folding(path)

    train, test = cv_loo(df_1, 5, 1)

    # X_train = normalized(np.hstack(test.T[0:29], test.T[34:54])).T

    X_train = normalized([np.hstack((row[0:29], row[34:54])) for row in train])
    X_test = normalized([np.hstack((row[0:29], row[34:54])) for row in test])

    Y_train = [row[29] for row in train]
    Y_train = np.expand_dims(Y_train, axis=-1)

    Y_test = [row[29] for row in test]
    Y_test = np.expand_dims(Y_test, axis=-1)

    lrg.train(1000, X_train, Y_train)
    lrg.validate(X_test, Y_test)


