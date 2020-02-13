from src.models import LinearRegressionWithGd
from src.common import *

# path = "/home/jan/Documents/gradient_descent_with_linear_regr/Dataset/Training/"
# path = "../Dataset/Training/"
path = "../Dataset/Training/"

if __name__ == "__main__":

    num_folds = 5
    epochs = 1000
    learning_rate = 1e-3

    lrg = LinearRegressionWithGd()
    lrg.learning_rate = learning_rate

    df_1, df_2 = folding(path)
    np.random.shuffle(df_1)

    stats = []
    for i in range(num_folds):
        train, test = cv_loo(df_1, num_folds, i)

        X_train = normalized(train.T[0:53].T)
        X_test = normalized(test.T[0:53].T)

        Y_train = train.T[53].T
        Y_train = np.expand_dims(Y_train, axis=-1)

        Y_test = test.T[53].T
        Y_test = np.expand_dims(Y_test, axis=-1)

        lrg.train(epochs, X_train, Y_train)
        stat = lrg.validate(X_test, Y_test)
        stats.append(stat)
        print(lrg.betas.T)
        print(lrg.bias)
    
    mean_stats = np.sum(stats, axis=0) / num_folds
    print("mse is {}, r2_val is {}, rmse is {}, r2/rmse {}".format(
        mean_stats[0], mean_stats[1], mean_stats[2], mean_stats[3]))


