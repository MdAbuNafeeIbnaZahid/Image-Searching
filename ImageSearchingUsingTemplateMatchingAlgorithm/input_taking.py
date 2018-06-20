import cv2
import os


def getRefTestImage(inputFileName):
    inputFileFullPath = getFullPathFromFileName(inputFileName)
    inputFile = open(inputFileFullPath, 'r')

    refFileName = inputFile.readline()[:-1]
    testFileName = inputFile.readline()[:-1]
    # readline returns a line with \n at the end

    refImage = getImageFromImageName(refFileName)
    testImage = getImageFromImageName(testFileName)

    return refImage, testImage


def getFullPathFromFileName(fileName):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fullPath = dir_path + '/' + fileName
    return fullPath


def getImageFromImageName(imageName):
    imageFullPath = getFullPathFromFileName(fileName=imageName)

    img = cv2.imread(imageFullPath, cv2.IMREAD_UNCHANGED)

    return img

