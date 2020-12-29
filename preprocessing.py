import PIL
import cv2
import glob
import os
import tqdm

dataTestPath = 'dataset\\test'
dataTrainPath = 'dataset\\train'

dataTestPathReformated = 'dataset\\testReformated'
dataTrainPathReformated = 'dataset\\trainReformated'
def displayDataInfo():
    xChanList = []
    yChanList = []
    for path in glob.glob(dataTestPath + '/*'):
        im = cv2.imread(path)
        y, x, z = im.shape
        yChanList.append(y)
        xChanList.append(x)

    for x in xChanList:
        if x != 80:
            print(x)
    print(max(yChanList))

    for path in glob.glob(dataTrainPath+'/*'):
        im = cv2.imread(path)
        y, x, z = im.shape
        yChanList.append(y)
        xChanList.append(x)

    for x in xChanList:
        if x != 80:
            print(x)
    print(max(yChanList))

def reformatDataToFixedSize():
    # Create new dirs
    os.mkdir(dataTrainPathReformated)
    os.mkdir(dataTestPathReformated)

    for path in tqdm.tqdm(glob.glob(dataTestPath + '/*')):
        im = cv2.imread(path)
        y, _, _ = im.shape
        replicate = cv2.copyMakeBorder(im, 0, 300 - y, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        cv2.imwrite(dataTestPathReformated + '\\' + path.split('\\')[-1], replicate)

    for path in tqdm.tqdm(glob.glob(dataTrainPath + '/*')):
        im = cv2.imread(path)
        y, _, _ = im.shape
        replicate = cv2.copyMakeBorder(im, 0, 300-y, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])

        cv2.imwrite(dataTrainPathReformated + '\\' + path.split('\\')[-1], replicate)

reformatDataToFixedSize()
