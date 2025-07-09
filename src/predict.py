
# predict.py
import joblib
import numpy as np
import os
from sklearn.datasets import load_iris

# Load model
model_path = os.path.join("model", "iris_model.pkl")
model = joblib.load(model_path)

# Load feature names from sklearn iris dataset
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Example input: [sepal_length, sepal_width, petal_length, petal_width]
# Order must match iris.feature_names
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict
prediction = model.predict(sample)
predicted_class = prediction[0]

print("Predicted class:", predicted_class)
