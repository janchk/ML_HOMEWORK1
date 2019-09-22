# from sklearn import regressor


def lin_regression(x, betas, eps=0):
    y = 0
    for beta in betas:
        y += beta*x
    y += eps
    return y



