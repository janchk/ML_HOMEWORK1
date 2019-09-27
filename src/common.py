import numpy as np
import os
import random as rnd
import pandas as pd


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