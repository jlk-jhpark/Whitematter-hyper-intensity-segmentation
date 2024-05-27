from scipy.stats import entropy
import numpy as np

def entropy_cal(img_sum):
    m_value = np.mean(img_sum,0)
    pk= [m_value,1-m_value]
    qk = [0.5,0.5]
    return 1-entropy(pk,qk, base=2)