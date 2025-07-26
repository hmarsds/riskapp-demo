import numpy as np
import riskfolio as rp

def MAD(x):
    x = np.asarray(x).flatten()
    m = x.mean()
    return np.mean(np.abs(x - m))


def VaR_Hist(x, alpha=0.05):
    return rp.RiskFunctions.VaR_Hist(np.asarray(x).flatten(), alpha)

def CVaR_Hist(x, alpha=0.05):
    return rp.RiskFunctions.CVaR_Hist(np.asarray(x).flatten(), alpha)

def EVaR_Hist(x, alpha=0.05):
    return rp.RiskFunctions.EVaR_Hist(np.asarray(x).flatten(), alpha)  # returns (evar, something)

def RLVaR_Hist(x, alpha=0.05):
    return rp.RiskFunctions.RLVaR_Hist(np.asarray(x).flatten(), alpha)

def TG(x, alpha=0.05):
    return rp.RiskFunctions.TG(np.asarray(x).flatten(), alpha)

def WR(x):
    x = np.asarray(x).flatten()
    return (x > 0).mean()  # proportion of positive returns

def LPM(x, MAR=0.0, p=1):
    x = np.asarray(x).flatten()
    diff = np.clip(MAR - x, 0, None)
    return np.mean(diff**p)