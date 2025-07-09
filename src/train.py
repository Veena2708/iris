
# train.py (auto-load Iris from sklearn)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Add class label
df["class"] = df["target"].map(dict(enumerate(iris.target_names)))

# Features and labels
X = df[iris.feature_names]
y = df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/iris_model.pkl")

# Accuracy
print("Training Accuracy:", clf.score(X_train, y_train))
print("Testing Accuracy:", clf.score(X_test, y_test))
