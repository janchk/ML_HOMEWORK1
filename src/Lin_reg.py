def lin_regression(x, betas, eps=0):
    # y = 0
    beta_0 = 0
    y = beta_0
    # y = sum(for beta in )
    for i in range(len(x[0])):
        y += betas[i]*x[i]
    return y




