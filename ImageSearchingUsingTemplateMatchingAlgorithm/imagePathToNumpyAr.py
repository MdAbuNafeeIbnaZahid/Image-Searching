"""
https://stackoverflow.com/a/34533139/3798217
"""


from PIL import Image
import numpy

def image2pixelarray(image):

    """
    Parameters
    ----------
    image : Pillow Image


    Returns
    -------
    a 3 dimensional numpy array
    """

    (width, height) = image.size
    greyscale_map = list(image.getdata())
    greyscale_map = numpy.array(greyscale_map)
    greyscale_map = greyscale_map.reshape((height, width, 3))
    return greyscale_map