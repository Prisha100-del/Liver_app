from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

# Create a simple test model
X = np.random.rand(100, 14)  # 14 features
y = np.random.randint(0, 2, 100)  # Binary classification

# Create and fit the model
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(X)

# Save the model and scaler
with open('liver_cirrhosis_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('liver_cirrhosis_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f) 