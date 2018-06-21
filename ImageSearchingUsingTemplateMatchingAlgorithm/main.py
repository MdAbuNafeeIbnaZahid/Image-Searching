from image_search_algo import *

from input_taking import getRefTestImage


inputFileName = 'input.txt'
testImage, refImage = getRefTestImage(inputFileName)


exImFinder = ExhaustiveImageFinder(testImage=testImage, referenceImage=refImage)
hierImFinder = HierarchicalImageFinder(testImage=testImage, referenceImage=refImage)
logImFinder = LogarithmicImageFinder(testImage=testImage, referenceImage=refImage)

matchedRectangle = logImFinder.findMatchedRectangle()

print( matchedRectangle )

rectangledImage = getRectangledImage(img=testImage, rectangle=matchedRectangle)
cv2.imshow('img', rectangledImage)
cv2.waitKey(0)
cv2.destroyAllWindows()