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

    def __sub__(self, other):
        ret = Point(self.w - other.w, self.h - other.h)
        return ret

    def getMin(self):
        return min(self.w, self.h)

    def __mul__(self, other):
        ret = Point(self.w * other, self.h*other)
        return ret

    def getWHTuple(self):
        return (self.w, self.h)

    def getElementWiseOperatedPoint(self, anotherPoint, operation):

        newW = operation( self.w, anotherPoint.w )
        newH = operation(self.h, anotherPoint.h)

        ret = Point( w=newW, h=newH  )
        return ret

    def getIntPoint(self):
        ret = Point( int(self.w), int(self.h) )
        return ret





class Rectangle(AttrDisplay):


    def __init__(self, topLeft, w, h):
        self.topLeft = topLeft
        self.bottomRight = topLeft + Point(w=w,h=h)


    def __add__(self, otherPoint):
        ret = copy.deepcopy(self)

        ret.topLeft += otherPoint
        ret.bottomRight += otherPoint

        return ret


    def __sub__(self, otherPoint):
        ret = copy.deepcopy(self)

        ret.topLeft -= otherPoint
        ret.bottomRight -= otherPoint

        return ret


    def getTopLeftTuple(self):
        return self.topLeft.getWHTuple()

    def getBottomRightTuple(self):
        return self.bottomRight.getWHTuple()

    def isInImage(self, img):
        iw, ih = getWH(img=img)

        if self.topLeft.getMin() < 0:
            return False
        if self.bottomRight.w > iw or self.bottomRight.h > ih:
            return False

        return True

    def getTopH(self):
        return self.topLeft.h

    def getBottomH(self):
        return self.bottomRight.h

    def getLeftW(self):
        return self.topLeft.w

    def getRightW(self):
        return self.bottomRight.w

    def getEnlargedRectangle(self, d):

        ret = copy.deepcopy(self)
        ret.topLeft += Point(-d,-d)
        ret.bottomRight += Point(d,d)

        return ret

    def getChoppedRectByImage(self, img):
        iw, ih = getWH(img=img)

        ret = copy.deepcopy(self)

        zeroZeroPoint = Point(0,0)
        ret.topLeft = ret.topLeft.getElementWiseOperatedPoint(anotherPoint=zeroZeroPoint, operation=max)

        whPoint = Point(w=iw,h=ih)
        ret.bottomRight = ret.bottomRight.getElementWiseOperatedPoint(anotherPoint=whPoint, operation=min)

        return ret

    def getSlidedRectInsideImage(self, img):
        ret = copy.deepcopy(self)

        iw, ih = getWH(img=img)

        zeroPoint = Point(0,0)
        shiftForZP = (zeroPoint - ret.topLeft)
        shiftForZP = shiftForZP.getElementWiseOperatedPoint(anotherPoint=zeroPoint, operation=max)

        whPoint = Point(iw,ih)
        shiftForWH = whPoint - ret.bottomRight
        shiftForWH = shiftForWH.getElementWiseOperatedPoint(anotherPoint=zeroPoint, operation=min)

        ret = ret + shiftForZP + shiftForWH
        return ret

def getBottomRightPoint(img):
    w,h = getWH(img=img)
    return Point(w,h)


def getRectangle(topLeft, refIm):
    w,h = getWH(refIm)
    ret = Rectangle(topLeft=topLeft, w=w, h=h)
    return ret



def getCroppedImage(img, rectangle):
    assert rectangle.isInImage(img)

    topH = rectangle.getTopH()
    bottomH = rectangle.getBottomH()

    leftW = rectangle.getLeftW()
    rightW = rectangle.getRightW()

    croppedImage = img[topH:bottomH, leftW:rightW]

    return croppedImage


def getBottomRight(topLeft, refIm):
    refW, refH = getWH(refIm)
    bottomRight = Point(topLeft.w + refW, topLeft.h + refH)

    return bottomRight



def getCroppedImageFromRefImTopLeft(testIm, refIm, topLeft):
    bottomRight = getBottomRight(topLeft=topLeft, refIm=refIm)

    croppedIm = testIm[topLeft.h : bottomRight.h, topLeft.w : bottomRight.w]
    return croppedIm


def getWH(img):
    h = img.shape[0]
    w = img.shape[1]
    return w,h

def isWithin(testIm, refIm, topLeft):

    testW, testH = getWH(testIm)
    bottomRight = topLeft + getBottomRightPoint(img=refIm)

    if topLeft.getMin() < 0:
        return False
    if bottomRight.w > testW or bottomRight.h > testH :
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

    croppedImage = getCroppedImageFromRefImTopLeft(testIm=testIm, refIm=refIm, topLeft=topLeftPoint)
    ret = getDifBetweenSameSizeImages(im1=refIm, im2=croppedImage)

    return ret


def getDifBetweenSameSizeImages(im1, im2):
    assert im1.shape == im2.shape

    difMap = getDifMap(im1, im2)
    ret = difMap[0][0]
    return ret


def getBestMatchLoc(testIm, refIm):
    difMap = getDifMap(testIm=testIm, refIm=refIm)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(difMap)
    minLoc = Point(minLoc[0], minLoc[1])
    return minLoc


def getLocInBig(testIm, refIm, posTopLeftInBig):

    minDif = float('inf')
    ret = None

    posRect = getRectangle(topLeft=posTopLeftInBig, refIm=refIm)
    # print( 'posRect = ', posRect )
    enlargedRect = posRect.getEnlargedRectangle(d=3)
    # print('enlargedRect = ' , enlargedRect)
    # print('testIm.shape = ', testIm.shape)
    validRectangleForCropping = enlargedRect.getChoppedRectByImage(img=testIm)
    # print('validRectangleForCropping = ', validRectangleForCropping )
    croppedTestIm = getCroppedImage(img=testIm, rectangle=validRectangleForCropping)
    # print( ' croppedTestIm.shape = ', croppedTestIm.shape  )

    bestMatchLocInEnlargedCropped = getBestMatchLoc(testIm=croppedTestIm, refIm=refIm)

    ret = bestMatchLocInEnlargedCropped + validRectangleForCropping.topLeft

    return ret



class ImageFinder(object):

    def __init__(self, testImage, referenceImage):
        self.testImage = testImage
        self.referenceImage = referenceImage

        self.refW, self.refH = getWH(referenceImage)
        self.testW, self.testH = getWH(testImage)


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
            return ret
        else:

            smallTestIm = cv2.pyrDown(testIm)
            smallRefIm = cv2.pyrDown(refIm)

            locInSmall = self.getUpperLeftMatchHier(smallTestIm, smallRefIm)
            # print('locInSmall = ', locInSmall)
            posLocInBig = locInSmall * 2
            # print('posLocInBig = ', posLocInBig)


            ret = getLocInBig(testIm=testIm, refIm=refIm, posTopLeftInBig=posLocInBig)
            return ret




    def findUpperLeftMatch(self):
        return self.getUpperLeftMatchHier(self.testImage, self.referenceImage)





class LogarithmicImageFinder(ImageFinder):
    dwAr = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    dhAr = [-1, 0, 1, -1, 0, 1, -1, 0, 1]

    def __init__(self, testImage, referenceImage):
        ImageFinder.__init__(self, testImage=testImage, referenceImage=referenceImage)
        self.refBottomRight = getBottomRightPoint(img=self.referenceImage)
        self.testBottomRight = getBottomRightPoint(img=self.testImage)

        self.p = self.refBottomRight * 0.5

        self.center = self.testBottomRight * 0.5 - self.refBottomRight * 0.5


    def getNineTopLeftPoints(self):
        ret = []
        for i in range(0,9):
            dw = LogarithmicImageFinder.dwAr[i] * self.p.w
            dh = LogarithmicImageFinder.dhAr[i] * self.p.h

            curTopLeft = self.center + Point(dw,dh)
            curTopLeft = curTopLeft.getIntPoint()
            curRectangle = getRectangle(curTopLeft, self.referenceImage)
            curValidRectangle = curRectangle.getSlidedRectInsideImage(self.testImage)

            curValidTopLeft = curValidRectangle.topLeft

            ret.append(curValidTopLeft)

        return ret


    def findUpperLeftMatch(self):

        while( self.p.getMin() > 1 ):
            nineTopLeftPoints = self.getNineTopLeftPoints()

            posCenter = None
            minCost = float('inf')
            for topLeftPoint in nineTopLeftPoints:
                curCost = getDif(testIm=self.testImage, refIm=self.referenceImage, topLeftPoint=topLeftPoint)
                if curCost < minCost:
                    minCost = curCost
                    posCenter = topLeftPoint

            self.center = posCenter
            self.p = self.p * 0.5

        return self.center

