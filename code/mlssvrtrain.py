import numpy as np
def MLSSVRTrain(trnX, trnY, gamma, lambd, p):
    trnY = trnY.reshape(trnY.shape[0], -1)

    # Check that the number of rows in trnX and trnY are equal
    if trnX.shape[0] != trnY.shape[0]:
        print('The number of rows in trnX and trnY must be equal.')
        return
    
    l, m = trnY.shape

    # Compute the RBF kernel matrix K
    K = np.exp(-gamma * np.sum((trnX[:, np.newaxis] - trnX[np.newaxis, :]) ** 2, axis=2))

    # Construct the matrix H
    H = np.tile(K, (m, m)) + np.eye(m * l) / gamma

    # Construct the matrix P
    P = np.zeros((m * l, m))
    for t in range(m):
        idx1 = l * t
        idx2 = l * (t + 1)

        H[idx1:idx2, idx1:idx2] += K * (m / lambd)

        P[idx1:idx2, t] = np.ones(l)

    # Solve for alpha and b
    eta = np.linalg.solve(H, P)
    nu = np.linalg.solve(H, trnY.reshape(-1, 1))
    S = np.dot(P.T, eta)
    b = np.linalg.solve(S, np.dot(eta.T, trnY.reshape(-1, 1)))
    alpha = (nu.reshape(l*m, 1) - np.dot(eta, b)).reshape(l,m)

    return alpha, b
