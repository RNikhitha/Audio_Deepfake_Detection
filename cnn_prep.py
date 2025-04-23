import os
import numpy as np
import pandas as pd
import cv2

X_train = []
y_train = []

df = pd.read_csv('data_annotated_features.csv')
df.drop_duplicates(subset=['file_name'], inplace=True)

df1 = df.copy()

for f in os.listdir("en_train"):
    if f.endswith('.png'):
        csv_name = f.replace('_melspec.png','.wav')
        img_path = os.path.join("en_train", f)
        img = cv2.imread(img_path)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = cv2.resize(img, (128, 128))  
        img = img / 255.0  

        X_train.append(img)

        y = df1.loc[df['file_name'] == csv_name, 'label_encoded']
        y_train.append(int(y))

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"X_test shape: {X_train.shape}")
print(f"y_test shape: {y_train.shape}")

np.save('X_test.npy', X_train)
np.save('y_test.npy', y_train)
