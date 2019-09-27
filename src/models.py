from stat_funcs import *
import numpy as np


class LinearRegressionWithGd:
    def __init__(self):
        self.betas = [] 
        self.bias = 0
        self.epoch = 0
        self.learning_rate = 0.05
        self.batch_size = 5

    def __regression(self, x):        
        if not len(self.betas):
            self.betas = np.expand_dims(np.random.rand(len(x[0])), axis=-1)
        else:
            pass
        return x.dot(self.betas) + self.bias
    
    def __create_minibatch(self, x, y):
        data = np.hstack((x, y))
        mini_batches = (np.array_split(data, self.batch_size))
        mini_batches = [(mb.T[0:53].T, np.expand_dims(mb.T[53].T, axis=-1)) for mb in mini_batches]
        return mini_batches

    def __minibatch_gradient_descent(self, x, y):
        mini_batches = self.__create_minibatch(x, y)
        for mini_batch in mini_batches:
            x_mini, y_mini = mini_batch
            y_pred_mini = self.__regression(x_mini)
            self.bias = self.bias - self.learning_rate * (mse_gradient_bias(y_mini, y_pred_mini))
            self.betas = self.betas - self.learning_rate * (mse_gradient_betas(x_mini, y_mini, y_pred_mini))

    def __gradient_descent(self, x, y, y_pred):
        self.bias = self.bias - self.learning_rate * (mse_gradient_bias(y, y_pred))
        self.betas = self.betas - self.learning_rate * (mse_gradient_betas(x, y, y_pred))

    def train(self, n_epochs, x, y):
        for epoch in range(n_epochs):
            y_pred = self.__regression(x)
            # self.__gradient_descent(x, y, y_pred)
            self.__minibatch_gradient_descent(x, y)
            print("train loss is {}".format(mse(y, y_pred)))

    def validate(self, x, y):
        y_pred = self.__regression(x)
        r2_val = R2(y, y_pred)
        mse_val = mse(y, y_pred)
        rmse_val = mse_val**(1/2)
        r2_rmse = r2_val/rmse_val
        print("val loss is {}, R2 is {}, RMSE is {}, R2/RMSE is {}".format(
            mse_val, r2_val, rmse_val, r2_rmse))
        return mse_val, r2_val, rmse_val, r2_rmse

