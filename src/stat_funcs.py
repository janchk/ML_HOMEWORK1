import math


def RMSE(ground_truth, list_of_vectors):
    RMSE_res = 0
    for vect in list_of_vectors:
        RMSE_res += (ground_truth - vect) ** 2

    RMSE_res = RMSE_res / len(list_of_vectors)
    RMSE_res = math.sqrt(RMSE_res)

    return RMSE_res


def MSE(ground_truth, list_of_vectors):
    MSE_res = 0
    for vect in list_of_vectors:
        MSE_res += (ground_truth - vect) ** 2

    MSE_res = MSE_res / len(list_of_vectors)

    return MSE_res


def MSE_gradient(X, ground_truth, prediction):
    N = len(X)

    return (1/N) * X.T.dot((prediction - ground_truth))

#
# def R2():
