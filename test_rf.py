from sklearn.ensemble import RandomForestRegressor
import numpy as np
X = np.array([[1], [2], [3]])
y = np.array(['NEAR OCEAN', 'INLAND', 'NEAR OCEAN'])
try:
    RandomForestRegressor().fit(X, y)
except Exception as e:
    print(repr(e))
