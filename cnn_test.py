import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Load model and test data 
model = load_model("image_classification.h5")
X_test = np.load("X_test.npy")
Y_test = np.load("y_test.npy")

probs = model.predict(X_test).flatten() 

pred_labels = (probs >= 0.5).astype(int)
accuracy = np.mean(pred_labels == Y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# EER calculation
fpr, tpr, _ = roc_curve(Y_test, probs)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
print(f"Equal Error Rate (EER) for CNN: {eer:.4f}")