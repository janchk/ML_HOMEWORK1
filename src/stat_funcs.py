import numpy as np


def mse(ground_truth, predictions):
    """

    :param ground_truth: vectors of GT values
    :param predictions: vectors of predicted values
    :return: Mean Square Error
    """
    return np.sum((ground_truth - predictions)**2)/len(predictions)


def R2(Y, predictions):
    mean_pred_value = np.mean(Y)
    SS_tot = np.sum((Y - mean_pred_value) ** 2)
    SS_res = np.sum((Y - predictions) ** 2)

    return 1 - (SS_res / SS_tot)


def mse_gradient_betas(X, ground_truth, prediction):
    """
    :param X: Features
    :param ground_truth:
    :param prediction:
    :return: derivative of MSE function
    """
    return (-2 / len(prediction)) * X.T.dot((ground_truth - prediction))


def mse_gradient_bias(ground_truth, prediction):
    return (-2 / len(prediction)) * np.ones((1, len(prediction))).dot((ground_truth - prediction))



