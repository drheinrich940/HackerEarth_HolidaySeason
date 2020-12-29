import pandas as pd
import os
from shutil import copyfile
import tqdm

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

sortData('dataset\\trainReformated', 'dataset\\train.csv', 'dataset\\trainSorted')