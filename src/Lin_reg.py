def lin_regression(x, betas, eps=0):
    y = 0
    for i, beta in enumerate(betas):
        y += beta*x[i]
    y += eps
    return y




