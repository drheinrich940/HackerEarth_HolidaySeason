import PIL
import cv2
import glob

dataTestPath = 'dataset\\test'
dataTrainPath = 'dataset\\train'
def displayDataInfo():
    for testPath in glob.glob(dataTestPath+'/*'):
        print(testPath)

displayDataInfo()