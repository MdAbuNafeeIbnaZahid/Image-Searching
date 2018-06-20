import numpy as np
import cv2
import os


imageName = 'trainBDFlag.jpeg'



def getFullPathFromImageName(imageName):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fullPath = dir_path + '/' + imageName
    return fullPath


img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
cropped = img[10:100, 10:100 ]


cv2.imshow('image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()