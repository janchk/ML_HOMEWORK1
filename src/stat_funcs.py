import numpy as np


def mse(ground_truth, predictions):
    """

    :param ground_truth: vectors of GT values
    :param predictions: vectors of predicted values
    :return: Mean Square Error
    """
    return np.mean((ground_truth - predictions)**2)


def r2(y, predictions):
    mean_pred_value = np.mean(y)
    ss_tot = np.sum((y - mean_pred_value) ** 2)
    ss_res = np.sum((y - predictions) ** 2)

    return 1 - (ss_res / ss_tot)


def mse_gradient_betas(x, ground_truth, prediction):
    """
    :param x: Features
    :param ground_truth:
    :param prediction:
    :return: derivative of MSE function
    """
    return (-2 / len(prediction)) * x.T.dot((ground_truth - prediction))


def mse_gradient_bias(ground_truth, prediction):
    return (-2 / len(prediction)) * np.ones((1, len(prediction))).dot((ground_truth - prediction))



