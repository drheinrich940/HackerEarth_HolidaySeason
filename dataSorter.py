import pandas as pd
import os
from shutil import copyfile

df = pd.read_csv('dataset/train.csv')
print(df.head())
print(df.Class.unique())
# Create class folder
if not os.path.exists('dataset/sortedTrain'):
    os.makedirs('dataset/sortedTrain')
    for item in df.Class.unique():
        os.makedirs('dataset/sortedTrain/' + item)

# Copy images to associated folder
for index, row in df.iterrows():
    copyfile('dataset/train/' + row['Image'], 'dataset/sortedTrain/' + row['Class'] + '/' + row['Image'])
