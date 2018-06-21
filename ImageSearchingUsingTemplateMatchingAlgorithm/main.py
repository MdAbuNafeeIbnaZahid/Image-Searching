from image_search_algo import *

from input_taking import getRefTestImage


inputFileName = 'input.txt'
testImage, refImage = getRefTestImage(inputFileName)


exImFinder = ExhaustiveImageFinder(testImage=testImage, referenceImage=refImage)
matchedRectangle = exImFinder.findMatchedRectangle()

