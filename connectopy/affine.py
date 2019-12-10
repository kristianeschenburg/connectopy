import numpy as np


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform



class OLS_MV(object):

    """
    Class to perform (multivariate) linear regression.

    Parameters:
    - - - - -
    include_intercept : bool
        if True, adds intercept column to x matrix
    """

    def __init__(self, include_intercept=True):

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
