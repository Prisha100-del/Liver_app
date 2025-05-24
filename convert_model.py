import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create a simple random forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Create some dummy data (14 features as per your form)
X = np.random.rand(100, 14)
y = np.random.randint(0, 2, 100)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, 'liver_cirrhosis_model.pkl')
joblib.dump(scaler, 'liver_cirrhosis_scaler.pkl')

print("Model and scaler saved successfully!") 