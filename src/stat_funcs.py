import math


def RMSE(ground_truth, list_of_vectors):
    RMSE_res = 0
    for vect in list_of_vectors:
        RMSE_res += (ground_truth - vect) ** 2

    RMSE_res = RMSE_res / len(list_of_vectors)
    RMSE_res = math.sqrt(RMSE_res)

    return RMSE_res


# list of vectors
def MSE(ground_truth, predictions):
    MSE_res = 0
    for pred in predictions:
        MSE_res += (ground_truth - pred) ** 2

    MSE_res = MSE_res / len(predictions)

    return MSE_res


def MSE_gradient(ground_truth, prediction):
    return (2/len(prediction)) * (ground_truth - prediction)
    # N = len(X)

    # return (1 / N) * X.T.dot((prediction - ground_truth))


def R2(Y, predictions):
    mean_pred_value = sum(predictions) / len(predictions)

    SS_tot = 0
    for i in range(len(predictions)):
        SS_tot += (Y[i] - mean_pred_value)**2

    SS_res = 0
    for i in range(len(predictions)):
        SS_res += (Y[i] - predictions[i])**2

    return 1 - (SS_res / SS_tot)
