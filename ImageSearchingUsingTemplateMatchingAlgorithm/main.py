from PIL import Image

img = Image.open('BDFlagWaving.jpeg')

area = (100, 100, 150, 150)
cropped_img = img.crop(area)

cropped_img.show()
cropped_img.save("cr.jpg")