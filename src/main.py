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

source = None


# @gen.coroutine
# def update(new_data):
#     """
#     Func to update source by adding new data
#     :param new_data:
#     """
#     source.stream(new_data)


# def visualize(data_dict):
#     source = ColumnarDataSource(
#         data=data_dict
#     )
#     plot = figure()
#     plot.line(x='epochs', y="R2", color="green", alpha=0.8, legend="R2", line_witdh=2, source=source)
#     doc = curdoc()
#     # Add the plot to the current document
#     doc.add_root(plot)
#     return doc, source


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_csv_data(fpath):
    df = pd.read_csv(fpath)
    return df.values


def folding(path):
    fs_iter = [path + fpath for fpath in os.listdir(path) if fpath.endswith(".csv")]
    train = get_csv_data(fs_iter[rnd.randint(0, 4)])
    test = get_csv_data(fs_iter[rnd.randint(0, 4)])

    return train, test


if __name__ == "__main__":

    lrg = LinearRegressionWithGd()

    df_array_train, df_array_test = folding(path)

    X_train = normalized([np.hstack((row[0:29], row[34:54])) for row in df_array_train])
    X_test = normalized([np.hstack((row[0:29], row[34:54])) for row in df_array_test])

    Y_train = [row[29] for row in df_array_train]
    Y_train = np.expand_dims(Y_train, axis=-1)

    Y_test = [row[29] for row in df_array_test]
    Y_test = np.expand_dims(Y_test, axis=-1)

    lrg.train(1000, X_train, Y_train)
    lrg.validate(X_test, Y_test)


