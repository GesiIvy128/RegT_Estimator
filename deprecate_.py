def RT(X:np.array,prev_mean:np.array=None,prev_cov:np.array=None,
       type:str='I',theta:float=1e-2,
       nu:int=3,tol:float=1e-4,max_iter:int=100):
    n_samples,n_features = X.shape
    S = psi_fun(X)['cov']
    # initial values of mean and cov
    prev_mean = np.mean(X,axis = 0) if prev_mean is None else prev_mean
    prev_cov = (0.1 * np.eye(n_features) + 0.9 * S) if prev_cov is None else prev_cov
    prev_precision = np.linalg.inv(prev_cov)

    iterations = 0
    while True:
        # E-step
        weights,MDs = [],[]
        for i in range(n_samples):
            part1 = nu + n_features
            MD = ((X[i, :] - prev_mean) @ prev_precision) @ (X[i, :] - prev_mean)
            part2 = nu + MD
            weights.append(part1 / part2)
            MDs.append(MD)
        # M-step
        psi = psi_fun(X,omega = np.array(weights))
        curr_mean,cov_ttype = psi['mean'],psi['cov']

        # Regularization process
        T = None
        if type == 'I':
            tau = np.trace(cov_ttype)
            T = tau / n_features * np.eye(n_features)
            curr_cov = theta * T + (1 - theta) * cov_ttype
            curr_precision = np.linalg.inv(curr_cov)
        elif type == 'EC':
            diag_cov = np.diag(cov_ttype)
            corr_ttype = cov_ttype / np.outer(np.sqrt(diag_cov),np.sqrt(diag_cov))
            tau = np.sum(corr_ttype)
            c = (tau - n_features) / (n_features ** 2 - n_features)
            T = c * np.ones((n_features,n_features)) + (1 - c) * np.eye(n_features)
            T = np.diag(np.sqrt(diag_cov)) @ T @ np.diag(np.sqrt(diag_cov))
            curr_cov = theta * T + (1 - theta) * cov_ttype
            curr_precision = np.linalg.inv(curr_cov)
        elif type == 'GL':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category = ConvergenceWarning)
                curr_cov,curr_precision = graphical_lasso(cov_ttype,alpha=theta)
        else:
            raise TypeError("Parameter 'type' Error")

        if (np.linalg.norm(curr_cov - prev_cov) < tol) or (iterations >= max_iter):
            break

        prev_mean,prev_cov,prev_precision = curr_mean,curr_cov,curr_precision
        iterations += 1
    out = {}
    out['mean'] = curr_mean
    out['cov'] = curr_cov
    out['precision'] = curr_precision
    out['weights'] = weights
    out['MDs'] = MDs
    out['theta'] = theta
    out['target'] = T
    out['iter'] = iterations + 1
    return out
