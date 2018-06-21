import cv2
import numpy as np
from collections import namedtuple
import copy
import time
from classtools import AttrDisplay


class Rectangle(AttrDisplay):

    def __init__(self, topLeft, w, h):
        self.topLeft = topLeft
        self.w = w
        self.h = h
        self.bottomRight = (topLeft[0]+w, topLeft[1]+h)

def getRectangledImage(img, rectangle):

    ret = copy.deepcopy(img)
    color = (255, 0, 0)
    thickness = 3

    cv2.rectangle(ret, rectangle.topLeft, rectangle.bottomRight, color=color, thickness=thickness )
    return ret


def getDifMap(testIm, refIm):
    difMap = cv2.matchTemplate(testIm, refIm, cv2.TM_SQDIFF)
    return difMap

def getDifBetweenSameSizeImages(im1, im2):
    assert im1.shape == im2.shape

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
        startTime = time.time()

        topLeftCorner = self.findUpperLeftMatch()

        endTime = time.time()
        elapsedTime = endTime-startTime
        print('elapsedTime = ' + str(elapsedTime) )

        rectangle = Rectangle(topLeft=topLeftCorner, w=self.refW, h=self.refH)

        return rectangle




class ExhaustiveImageFinder(ImageFinder):
    def findUpperLeftMatch(self):
        loc = getBestMatchLoc(self.testImage, self.referenceImage)
        return loc


class HierarchicalImageFinder(ImageFinder):
    def getUpperLeftMatchHier(self, testIm, refIm):
        refShape = refIm.shape
        minRefWOrH =  min(refShape[:2] )

        if (minRefWOrH < 10):
            return getBestMatchLoc(testIm=testIm, refIm=refIm)
        else:

            smallTestIm = cv2.pyrDown(testIm)
            smallRefIm = cv2.pyrDown(refIm)

            locInSmall = self.getUpperLeftMatchHier(smallTestIm, smallRefIm)
            ret = ( locInSmall[0]*2, locInSmall[1]*2 )

            return ret




    def findUpperLeftMatch(self):
        return self.getUpperLeftMatchHier(self.testImage, self.referenceImage)
