from src.Lin_reg import lin_regression
from src.stat_funcs import MSE_gradient, MSE


def gradient_desc(X, Y, betas,  learning_rate=0.1, iterations_number=1000):

    predictions = []
    for it in range(iterations_number):
        prediction = lin_regression(X, betas)
        # predictions.append(prediction)

        betas = betas - learning_rate * MSE_gradient(X, Y, prediction)
        print("iteration is {} error is {}, Y is {}, pred is {}".format(it, MSE(Y, prediction), Y[2], prediction[2]))
        pass

    return betas

# if __name__ == "__main__":
