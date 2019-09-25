from src.stat_funcs import *


class LinearRegressionWithGd:
    def __init__(self):
        self.betas = np.expand_dims(np.random.rand(49), axis=-1)
        self.bias = 0
        self.epoch = 0
        self.learning_rate = 0.05

    def __regression(self, x):
        return x.dot(self.betas) + self.bias

    def __gradient_descent(self, x, y, y_pred):
        self.bias = self.bias - self.learning_rate * (mse_gradient_bias(y, y_pred))
        self.betas = self.betas - self.learning_rate * (mse_gradient_betas(x, y, y_pred))

    def train(self, n_epochs, x, y):
        for epoch in range(n_epochs):
            y_pred = self.__regression(x)
            self.__gradient_descent(x, y, y_pred)
            print("train loss is {}".format(mse(y, y_pred)))

    def validate(self, x, y):
        y_pred = self.__regression(x)
        r2_val = R2(y, y_pred)
        mse_val = mse(y, y_pred)
        rmse_val = mse_val**(1/2)
        print("val loss is {}, R2 is {}, RMSE is {}, R2/RMSE is {}".format(
            mse_val, r2_val, rmse_val, r2_val/rmse_val))

