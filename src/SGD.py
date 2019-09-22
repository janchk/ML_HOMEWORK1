import numpy as np

from src.Lin_reg import lin_regression
from src.stat_funcs import MSE_gradient


def gradient_desc(X, betas, ground_truth, learning_rate=0.01, iterations_number=100):

    for it in range(iterations_number):
        prediction = lin_regression(X, betas)

        betas = betas - learning_rate * MSE_gradient(X, ground_truth, prediction)

    return betas