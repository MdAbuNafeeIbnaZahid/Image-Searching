from PIL import Image, ImageFilter, ImageDraw
from imagePathToNumpyAr import image2pixelarray
import copy
from numpy import linalg as LA



trainImageFilePath = 'trainBDFlag.jpeg'
train = Image.open(trainImageFilePath)
(width, height) = train.size


draw = ImageDraw.Draw(train)
draw.rectangle( ( (0,0), (100,100) ) )

train.show()


newTrain = train.filter( ImageFilter.BLUR )

trainAr = image2pixelarray(train)
newTrainAr = image2pixelarray(newTrain)


print( LA.norm(trainAr) )
