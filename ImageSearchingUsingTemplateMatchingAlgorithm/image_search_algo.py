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
    minVal, maxVal, minLoc, maxLoc = cv2.MinMaxLoc(difMap)
    return minLoc




class ImageFinder(object):

    def __init__(self, testImage, referenceImage):
        self.testImage = testImage
        self.referenceImage = referenceImage

    def findImage(self):
        """
        This return tuples of two paris. first pair is the upper-left corner.
        second pair is the lower right corner
        """
        raise NotImplemented







class ExhaustiveImageFinder(ImageFinder):
    def findImage(self):
        loc = getBestMatchLoc(self.testImage)
        return loc