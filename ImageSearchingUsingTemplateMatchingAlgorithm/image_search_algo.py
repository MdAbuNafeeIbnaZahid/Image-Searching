import cv2
import numpy as np
from collections import namedtuple
import copy
import time
from classtools import AttrDisplay


class Point(AttrDisplay):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __add__(self, other):
        ret = Point(self.w + other.w, self.h + other.h)
        return ret

    def getMin(self):
        return min(self.w, self.h)

    def __mul__(self, other):
        ret = Point(self.w * other, self.h*other)
        return ret

    def getWHTuple(self):
        return (self.w, self.h)

class Rectangle(AttrDisplay):
    def __init__(self, topLeft, w, h):
        self.topLeft = topLeft
        self.w = w
        self.h = h
        self.bottomRight = topLeft + Point(w=w,h=h)

    def getTopLeftTuple(self):
        return self.topLeft.getWHTuple()

    def getBottomRightTuple(self):
        return self.bottomRight.getWHTuple()

def getWH(img):
    h = img.shape[0]
    w = img.shape[1]
    return w,h

def isWithin(testIm, refIm, topLeft):

    refW, refH = getWH(refIm)
    testW, testH = getWH(testIm)
    bottomRight = topLeft + Point(refW, refH)

    if topLeft.getMin() < 0:
        return False
    if bottomRight.w > testW or bottomRight.h > tesH :
        return False

    return True


def getRectangledImage(img, rectangle):
    ret = copy.deepcopy(img)
    color = (255, 0, 0)
    thickness = 3

    cv2.rectangle(ret, rectangle.getTopLeftTuple(), rectangle.getBottomRightTuple(), color=color, thickness=thickness )
    return ret


def getDifMap(testIm, refIm):
    difMap = cv2.matchTemplate(testIm, refIm, cv2.TM_SQDIFF)
    return difMap


def getDif(testIm, refIm, topLeftPoint):
    if not isWithin(testIm=testIm, refIm=refIm, topLeft=topLeftPoint):
        return float('inf')


def getDifBetweenSameSizeImages(im1, im2):
    assert im1.shape == im2.shape

    difMap = getDifMap(im1, im2)
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

        self.refW, self.refH = getWH(referenceImage)


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
            ret = getBestMatchLoc(testIm=testIm, refIm=refIm)
            return Point(w=ret[0], h=ret[1])
        else:

            smallTestIm = cv2.pyrDown(testIm)
            smallRefIm = cv2.pyrDown(refIm)

            locInSmall = self.getUpperLeftMatchHier(smallTestIm, smallRefIm)
            posLocInBig = locInSmall * 2


            ret = posLocInBig

            return ret




    def findUpperLeftMatch(self):
        return self.getUpperLeftMatchHier(self.testImage, self.referenceImage)
