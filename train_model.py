import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load the data
df = pd.read_csv('cleaned_cirrhosis.csv')

# Convert Sex to numeric (F=0, M=1)
df['Sex'] = df['Sex'].map({'F': 0, 'M': 1})

# Convert Yes/No to 1/0
binary_columns = ['Ascites', 'Hepatomegaly', 'Spiders']
for col in binary_columns:
    df[col] = df[col].map({'Y': 1, 'N': 0})

# Convert Edema to numeric (N=0, S=1, Y=2)
df['Edema'] = df['Edema'].map({'N': 0, 'S': 1, 'Y': 2})

# Select features and target
features = ['Age', 'Sex', 'Albumin', 'Bilirubin', 'Cholesterol', 'Copper', 
            'Alk_Phos', 'SGOT', 'Platelets', 'Prothrombin', 'Ascites', 
            'Hepatomegaly', 'Spiders', 'Edema']

# Convert Stage to binary (0: no cirrhosis, 1: cirrhosis)
# Assuming stages 3 and 4 indicate cirrhosis
df['Cirrhosis'] = (df['Stage'] >= 3).astype(int)
target = 'Cirrhosis'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the features
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Training accuracy: {train_score:.2f}")
print(f"Test accuracy: {test_score:.2f}")

# Save the model and scaler
with open('liver_cirrhosis_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('liver_cirrhosis_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f) 