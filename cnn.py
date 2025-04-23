from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


X_eval = np.load("X_eval.npy")
y_eval = np.load("y_eval.npy")
lb = LabelBinarizer()
y_train = lb.fit_transform(y_eval)

X_train, X_val, y_train, y_val = train_test_split(
    X_eval, y_train, test_size=0.2, random_state=42, stratify=y_train
)

num_classes = 2  

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train,  
    y_train,  
    epochs=1,  
    batch_size=32,  
    validation_data=(X_val, y_val),  
    verbose=1
)

#model.save('image_classification.h5')

val_loss, val_accuracy = model.evaluate(X_val, y_val)

print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

y_val_probabilities = model.predict(X_val)[:, 1]  

fpr, tpr, _ = roc_curve(y_val[:, 1], y_val_probabilities)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

print(f'Equal Error Rate (EER): {eer:.4f}')
