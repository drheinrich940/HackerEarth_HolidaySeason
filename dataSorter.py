import pandas as pd
import os
from shutil import copyfile
import tqdm
import random

def sortData(srcDir, srcCSV, destDir):
    df = pd.read_csv(srcCSV)
    print(df.head())
    print(df.Class.unique())
    # Create class folder
    if not os.path.exists(destDir):
        os.makedirs(destDir)
        for item in df.Class.unique():
            os.makedirs(destDir + '\\' + item)

    # Copy images to associated folder
    for index, row in tqdm.tqdm(df.iterrows()):
        copyfile(srcDir + '\\' + row['Image'], destDir + '\\' + row['Class'] + '\\' + row['Image'])

def sortDataForAugmentation(srcDir, srcCSV, destDirTrain, destDirVal, validation_split=0.2):
    df = pd.read_csv(srcCSV)
    print(df.head())
    print(df.Class.unique())
    # Create class folder
    if not os.path.exists(destDirTrain):
        os.makedirs(destDirTrain)
        for item in df.Class.unique():
            os.makedirs(destDirTrain + '\\' + item)

    if not os.path.exists(destDirVal):
        os.makedirs(destDirVal)
        for item in df.Class.unique():
            os.makedirs(destDirVal + '\\' + item)

    # Copy images to associated folder
    for index, row in tqdm.tqdm(df.iterrows()):
        if random.uniform(0, 1) > validation_split :
            copyfile(srcDir + '\\' + row['Image'], destDirTrain + '\\' + row['Class'] + '\\' + row['Image'])
        else:
            copyfile(srcDir + '\\' + row['Image'], destDirVal + '\\' + row['Class'] + '\\' + row['Image'])



# sortData('dataset\\trainReformated', 'dataset\\train.csv', 'dataset\\trainSorted')
sortDataForAugmentation('dataset\\trainReformated', 'dataset\\train.csv', 'dataset\\augmented\\trainAugmented',  'dataset\\augmented\\validationAugmented')