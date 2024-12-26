# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=";")

# Prepare features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Convert the target into a binary classification (e.g., Good or Bad wine)
y = y.apply(lambda x: 1 if x >= 6 else 0)  # Quality >= 6 is "Good"

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")

# Save the model as a pickle file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
