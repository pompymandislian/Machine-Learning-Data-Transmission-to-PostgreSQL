import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Generate dummy data
data = {
    'transaction': np.random.randint(1_000_000, 100_000_000, size=100),
    'age': np.random.randint(18, 65, size=100),
    'tenure': np.random.randint(0, 10, size=100),
    'num_pages_visited': np.random.randint(1, 20, size=100),
    'has_credit_card': np.random.choice([True, False], size=100),
    'items_in_cart': np.random.randint(0, 10, size=100),
    'purchase': np.random.choice([True, False], size=100)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into features and target
X = df.drop('purchase', axis=1)
y = df['purchase']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Scale numeric features
scaler = StandardScaler()
numeric_features = ['num_pages_visited', 'items_in_cart', 'transaction']
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Initialize and train logistic regression model
model = LogisticRegression(random_state=random_seed)
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model to disk
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)
