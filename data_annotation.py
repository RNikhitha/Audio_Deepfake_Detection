import pandas as pd
from sklearn.preprocessing import LabelEncoder

# dataset load
df_train = pd.read_csv('extracted_audio_features_test.csv')
df_dev = pd.read_csv('extracted_audio_features_dev.csv')
df_test = pd.read_csv('extracted_audio_features_eval.csv')

# labeling each wav 
with open('release_in_the_wild/meta.csv', 'r') as file:
    lines = file.readlines()

with open('en_dev.txt', 'r') as file:
    lines_dev = file.readlines()

with open('en_eval.txt', 'r') as file:
    lines_eval = file.readlines()

def add_labels(text, lines):
  all_lines = []
  for i in lines:
    all_lines.append(i.split(','))
  
  for i in all_lines:
    if text == i[0]:
      return i[-1].strip()

df_train['spoof_label'] = df_train['file_name'].apply(lambda x: add_labels(x, lines))
df_dev['spoof_label'] = df_dev['file_name'].apply(lambda x: add_labels(x, lines_dev))
df_test['spoof_label'] = df_test['file_name'].apply(lambda x: add_labels(x, lines_eval))
df_final = pd.concat([df_train, df_dev, df_test])   

# one hot encoding
# bonafide->0 , spoof->1
df_final = df_train.copy()
le = LabelEncoder()
df_final['label_encoded'] = le.fit_transform(df_final['spoof_label'])

df_final.to_csv('data_annotated_features_test.csv', index=False)