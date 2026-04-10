import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def normalize_data(X):
    """Normalize features using standard scaling."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def smooth_signal(signal, window_size=5):
    """Simple moving average smoothing."""
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def apply_smoothing(X, window_size=5):
    """Apply smoothing to each light curve."""
    return np.array([smooth_signal(x, window_size) for x in X])

def balance_data(X, y):
    """Balance dataset using SMOTE."""
    sm = SMOTE()
    return sm.fit_resample(X, y)
