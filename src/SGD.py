# from src.Lin_reg import lin_regression
from src.stat_funcs import *
import numpy as np
from sklearn.linear_model import LinearRegression


def lin_regression(x, betas, bias):
    y = x.dot(betas) + bias
    return y


def gradient_desc(X, Y, betas, bias=0, learning_rate=0.1, iterations_number=1000):

    # predictions = []
    for it in range(iterations_number):
        prediction = lin_regression(X, betas, bias)
        MSE = mse(Y, prediction)
        print("metrics is: RMSE {}, R2 {}".format(MSE**(1/2), R2(Y, prediction)))
        # predictions.append(prediction)

        bias = bias - learning_rate * (mse_gradient_bias(Y, prediction))
        betas = betas - learning_rate * mse_gradient_betas(X,  Y, prediction)
        print("iteration is {} error is {}, Y is {}, pred is {}, bias is {}".format(it, MSE, Y[2], prediction[2], bias))

    return betas

# if __name__ == "__main__":
