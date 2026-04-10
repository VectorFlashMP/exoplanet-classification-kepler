import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import normalize_data, apply_smoothing, balance_data
from models import train_mlp, evaluate_model

# Dummy placeholder (replace with real data loading)
X = np.random.rand(100, 100)  # 100 light curves
y = np.random.randint(0, 2, 100)

# Preprocessing
X = apply_smoothing(X)
X, scaler = normalize_data(X)
X, y = balance_data(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = train_mlp(X_train, y_train)

# Evaluate
acc, report = evaluate_model(model, X_test, y_test)

print("Accuracy:", acc)
print(report)
