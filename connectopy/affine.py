import numpy as np

class OLS_MV(object):

    """
    Class to perform (multivariate) linear regression.

    Parameters:
    - - - - -
    include_intercept : bool
        if True, adds intercept column to x matrix
    """

    def __init__(self, include_intercept):

        self.intercept = include_intercept

    def fit(self, x, y):

        """
        Parameters:
        - - - - - -
        x : float, array
            N x K regressor array
        y : float, array
            N x M response matrix
        """

        if x.ndim == 1:
            x = x[:, np.newaxis]

        if y.ndim == 1:
            y = y[:, np.newaxis]

        # get number of features including intercept term
        n, p = x.shape

        # construct design matrix with intercept column
        if self.intercept:
            intercept = np.ones((n))
            x = np.column_stack([intercept, x])

        # least squares model fitting
        # compute coefficient matrix, residuals, rank, and singular values
        # of least squares fit
        [A, _, rank, _] = np.linalg.lstsq(x, y, rcond=1)

        SSY = self.ssy(x, y)
        SSR = self.ssr(x, y)
        SSE = self.sse(x, y)

        # estimate multivariate variance
        sigma_hat = SSE/(n-rank-1)

        # root mean squared error of fitted values
        SSe = np.diag(SSE).sum()
        SSr = np.diag(SSR).sum()
        SSy = np.diag(SSY).sum()

        errors = np.sqrt(SSe/y.shape[0])

        rsquared = 1-(SSe/SSy)

        self.coefficients_ = A
        self.errors_ = errors
        self.r2_ = rsquared
        self.sigma_hat_ = sigma_hat
        self.sse_ = SSe
        self.ssr_ = SSr
        self.ssy_ = SSy
        self.fitted = True
    
    def predict(self, x):

        """
        Predict the response variable from a fitted model.

        Parameters:
        - - - - -
        x : float, array
            N x K regressor array
        """

        n, _ = x.shape

        if not self.fitted:
            raise('Model must be fitted first.')

        if self.intercept:
            intercept = np.ones((n))
            x = np.column_stack([intercept, x])
        
        y_pred = np.dot(x, self.coefficients_)
        return y_pred

    def ssr(self, x, y):

        """
        Compute the regression sum of squares.

        Parameters:
        - - - - -
        x: feature matrix
        y: response matrix
        """

        J = np.ones((y.shape[0], y.shape[0]))
        G = np.linalg.inv(np.dot(x.T, x))
        H = np.dot(x, np.dot(G, x.T))

        Qr = H - (1/y.shape[0])*J

        SSR = np.dot(y.T, np.dot(Qr, y))

        return SSR

    def sse(self, x, y):

        """
        Compute sum of squared errors.

        Parameters:
        - - - - -
        x: feature matrix
        y: response matrix
        """

        G = np.linalg.inv(np.dot(x.T, x))
        H = np.dot(x, np.dot(G, x.T))

        Qe = np.eye(y.shape[0]) - H

        SSE = np.dot(y.T, np.dot(Qe, y))

        return SSE

    def ssy(self, x, y):

        """
        Compute total sum of squares.

        Parameters:
        - - - - -
        x: feature matrix
        y: response matrix
        """

        J = np.ones((y.shape[0], y.shape[0]))
        Qt = np.eye(y.shape[0]) - (1/y.shape[0])*J

        SST = np.dot(y.T, np.dot(Qt, y))

        return SST


class SUR(object):

    """
    Class to perform Seemingly Unrelated Regression, using an implementation
    of feasible generalized least squares (FGLS).

    Parameters:
    - - - - -
    x : float, array
        N x K regressor array
    y : float, array
        N x 1 response vector
    """

    def __init__(self, include_intercept=True):

        """
        
        """

        self.intercept=include_intercept

    def fit(self, X, y):

        """
        Fit the SUR model.

        Parameters:
        - - - - -
        X : float, array
            data matrix of stacked linear equations
        y : float, array
            matrix of stacked response variables for each linear equation
        """

        if self.intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        beta_hat = self.fgls(X, y)

        return beta_hat


    def fgls(self, X, y):

        """
        Feasible generalized least squares (FGLS) algorithm.
        """

        # OLS step to compute residuals
        [beta, _, _, _] = np.linalg.lstsq(X, y, rcond=None)

        res = self.residuals(X, beta, y)

        # Generalized Least Squares to compute coefficients
        # and covariance matrix

        sigma = (1/res.shape[0])*np.outer(res, res)
        print(sigma)
        prec = np.linalg.inv(sigma)
        print(sigma)
        omega = np.kron(prec, np.eye(res.shape[0]))

        beta_hat = gls(X, y, omega)

        return [beta_hat, sigma]

    def residuals(self, X, y, beta):

        """
        Compute residuals of linear model.
        """

        return y - np.dot(X, beta)


def gls(X, y, omega):

    """
    Run the generalized least squares algorithm.

    beta^ = (X^{T} * Omega, X)^{-1} X^{T}*Omega*Y

    Parameters:
    - - - - -
    X : float, array
        data matrix
    y : float, array
        response matrix
    omega : float array
        inverse covariance matrix
    """

    xR = np.dot(X.T, omega)
    G = np.linalg.inv(np.dot(xR, X))

    beta = np.linalg.dot(G, np.dot(xR, y))

    return beta
