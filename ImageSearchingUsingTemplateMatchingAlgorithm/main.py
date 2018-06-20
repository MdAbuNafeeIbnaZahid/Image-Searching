from image_search_algo import *

from input_taking import getRefTestImage


inputFileName = 'input.txt'
refImage, testImage = getRefTestImage(inputFileName)


cv2.imshow('image', refImage)
cv2.waitKey(0)
cv2.destroyAllWindows()