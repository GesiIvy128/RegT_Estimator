# Author: Rui Zhang <zhrui1993@gmail.com>
#
# MIT License
import numpy as np

# Function to calculate the mean and covariance matrix of input data Y,
# optionally weighted by omega.
def psi_fun(Y: np.array, omega: np.array = None):
    # Initialize an empty covariance matrix with zeros
    psi = np.zeros((Y.shape[1], Y.shape[1]))

    # If omega (weights) is not provided, calculate a simple mean of Y along axis 0 (column-wise)
    if omega is None:
        Y_mean = np.mean(Y, axis=0)  # Calculate mean without weights

    # Loop through each row in Y
    for i in range(Y.shape[0]):
        if omega is None:
            # Update covariance matrix for unweighted data
            psi = np.outer(Y[i, :] - Y_mean, Y[i, :] - Y_mean) + psi
        else:
            # Calculate weighted mean if omega is provided
            Y_mean = np.average(Y, axis=0, weights=omega)
            # Update covariance matrix with weights
            psi = omega[i] * np.outer(Y[i, :] - Y_mean, Y[i, :] - Y_mean) + psi

    # Normalize by the number of observations to obtain the average covariance
    psi = psi / Y.shape[0]

    # Return the mean and covariance matrix as a dictionary
    return {
        'mean': Y_mean,  # Mean of the dataset Y
        'cov': psi       # Covariance matrix of the dataset Y
    }

# Function to calculate the log-likelihood score given precision and covariance matrices,
# with optional L1 regularization controlled by alpha.
def log_likelihood_score(precision: np.array, covariance: np.array, alpha: float = None):
    # Calculate the log determinant of the precision matrix
    log_det = np.linalg.slogdet(precision)[1]

    # Compute the log-likelihood score without regularization if alpha is None
    if alpha is None:
        log_likelihood = log_det - np.trace(np.dot(covariance, precision))
    else:
        # If alpha is provided, apply L1 regularization on the precision matrix
        log_likelihood = log_det - np.trace(np.dot(covariance, precision)) - alpha * np.linalg.norm(precision, ord=1)

    # Return the calculated log-likelihood score
    return log_likelihood
