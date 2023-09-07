import numpy as np
from mlssvrtrain import MLSSVRTrain
from mlssvrpredict import MLSSVRPredict
from tqdm import tqdm

def GridMLSSVR(trnX, trnY, fold):
    # Initialization
    gamma = np.power(2, np.arange(-1, 5, 2, dtype=float))
    lambd = np.power(2, np.arange(-1, 4, 2, dtype=float))
    p = np.power(2, np.arange(-4, 2, 2, dtype=float))
    m = trnY.shape[1]
    trnY = trnY[:, np.newaxis]

    # Random permutation
    indices = np.random.permutation(trnX.shape[0])
    trnX, trnY = trnX[indices], trnY[indices]

    MSE_best = np.inf
    MSE = np.zeros((fold, m))
    curR2 = np.zeros(m)
    R2 = np.zeros(m)

    # Grid search
    for i in tqdm(range(len(gamma))):
        for j in range(len(lambd)):
            for k in range(len(p)):
                predictY = np.empty((0, m))
                
                for v in range(fold):
                    # Cross-validation
                    start = v * trnX.shape[0] // fold
                    end = (v + 1) * trnX.shape[0] // fold
                    test_inst, test_lbl = trnX[start:end], trnY[start:end]
                    train_inst = np.concatenate((trnX[:start], trnX[end:]), axis=0)
                    train_lbl = np.concatenate((trnY[:start], trnY[end:]), axis=0)

                    alpha, b = MLSSVRTrain(train_inst, train_lbl, gamma[i], lambd[j], p[k])
                    tmpY, MSE[v, :],R2 = MLSSVRPredict(test_inst, test_lbl, train_inst, alpha, b, lambd[j], p[k])
                    predictY = np.concatenate((predictY, tmpY.reshape(-1, predictY.shape[1])), axis=0)


                curMSE = np.sum(MSE) / np.prod(trnY.shape)
                
                # Update best parameters
                if MSE_best > curMSE:
                    gamma_best, lambd_best, p_best, MSE_best,predictY_best = gamma[i], lambd[j], p[k], curMSE,predictY
    
    return gamma_best, lambd_best, p_best, MSE_best, predictY_best
