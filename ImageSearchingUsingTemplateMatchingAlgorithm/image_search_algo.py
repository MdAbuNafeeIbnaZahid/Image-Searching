import cv2
import numpy as np


def getDifMap(testIm, refIm):
    difMap = cv2.matchTemplate(testIm, refIm, cv2.TM_SQDIFF)
    return difMap

def getDifBetweenSameSizeImages(im1, im2):
    size1 = im1.shape
    size2 = im2.shape
    assert size1 == size2

    difMap = getDifMap
    ret = difMap[0][0]
    return ret


def getBestMatchLoc(testIm, refIm):
    difMap = getDifMap(testIm=testIm, refIm=refIm)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(difMap)
    return minLoc




class ImageFinder(object):

    def __init__(self, testImage, referenceImage):
        self.testImage = testImage
        self.referenceImage = referenceImage

        self.calculateRefShape()

    def calculateRefShape(self):
        self.refShape = self.referenceImage.shape
        self.refH = self.refShape[0]
        self.refW = self.refShape[1]

    def findUpperLeftMatch(self):
        """
        This returns a pair (row, column) of upper left of the image
        """
        raise NotImplemented


    def findMatchedRectangle(self):
        """
        :return: two pairs. first pair is the upper-left corner point
        second pair is the lower right corner point
        """
        upperLeftCorner = self.findUpperLeftMatch()

        upperRow = upperLeftCorner[1]
        leftCol = upperLeftCorner[0]

        raise NotImplemented




class ExhaustiveImageFinder(ImageFinder):
    def findUpperLeftMatch(self):
        loc = getBestMatchLoc(self.testImage, self.referenceImage)
        return loc