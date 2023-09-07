import numpy as np
from kerfun import Kerfun

def MLSSVRPredict(tstX, tstY, trnX, alpha, b, lmbda, p):
    """
    predictY, TSE, R2 = MLSSVRPredict(tstX, tstY, trnX, alpha, b, lmbda, p)

    tstX   - test data matrix (n x p)
    tstY   - test target matrix (n x m)
    trnX   - training data matrix (l x p)
    alpha  - alpha matrix (l x m)
    b      - bias vector (1 x m)
    lmbda  - regularization parameter
    p      - kernel parameter

    predictY - predicted output matrix (n x m)
    TSE      - total squared error vector (1 x m)
    R2       - squared correlation coefficient vector (1 x m)
    """
    tstY = tstY.reshape(tstY.shape[0],-1)
    if tstY.shape[1] != b.size:
        print('The number of columns in tstY and b must be equal.')
        return None, None, None

    l, m = alpha.shape

    if alpha.shape != (l, m):
        print(f"The size of alpha should be {l} x {m}")
        return None, None, None

    tstN = tstX.shape[0]
    b = b.reshape(-1, 1)

    K = Kerfun('rbf', tstX, trnX, p, 0)
    predictY = np.tile(np.sum(K.dot(alpha), axis=1), (m, 1)).T + K.dot(alpha) * (m / lmbda) + np.tile(b.T, (tstN, 1))



    # calculate Total Squared Error and squared correlation coefficient
    TSE = np.zeros(m)
    R2 = np.zeros(m)
    for t in range(m):
        TSE[t] = np.sum((predictY[:, t] - tstY[:, t])**2)
        R = np.corrcoef(predictY[:, t], tstY[:, t])
        if R.shape[0] > 1:
            R2[t] = R[0, 1]**2

    return predictY, TSE, R2
