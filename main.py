import skimage.io
import skimage.draw
import cv2
import numpy as np
import pandas as pd
from google.colab.patches import cv2_imshow
#import image
from matplotlib import pyplot as plt
# read input image
image = skimage.io.imread("/content/drive/MyDrive/images/alf.JPG")
#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')
plt.show()
gray = cv2.medianBlur(gray,5)
# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 127, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
plt.imshow(thresh_color)
plt.show()

thresh[thresh == 0] = 0
thresh[thresh == 255] = 1
height, width = thresh.shape
print(width)
print(height)
# Sum the value lines
vertical_px = np.sum(thresh, axis=0)
horozonital_px=np.sum(thresh,axis=1)
# Normalize
normalize = vertical_px/255
# create a black image with zeros
blankImage = np.zeros_like(thresh)
blankImage2 = np.zeros_like(thresh)
# Make the vertical projection histogram
for idx, value in enumerate(vertical_px):
    cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255,255,255), 1)
plt.imshow(blankImage)
plt.show()
for idx, value in enumerate(horozonital_px):
    cv2.line(blankImage2, (0, idx), (width-int(value),idx ), (255,255,255), 1)
plt.imshow(blankImage2)
plt.show()

# Concatenate the image
img_concate = cv2.vconcat(
    [image,  cv2.cvtColor(blankImage, cv2.COLOR_BGR2RGB)])
plt.imshow(img_concate)
plt.show()
blankImage2 = np.zeros_like(thresh)
for idx , value in enumerate(vertical_px):
    if value<11:
      new=cv2.line(thresh_color,(idx,0),(idx,height),(0,0,0))
plt.imshow(new)
plt.show()
gray_new=cv2.cvtColor(new,cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_new)

ret,thresh2=cv2.threshold(gray_new,50,255,0)
ret,thresh5=cv2.threshold(new,127,255,0,cv2.THRESH_BINARY+cv2.THRESH_BINARY_INV)
cv2_imshow(thresh5)
kernel2 = np.ones((5,5), np.uint8)
dilate = cv2.dilate(thresh_color, kernel2 ,iterations =2)
cv2_imshow(dilate)
dilate=cv2.cvtColor(dilate,cv2.COLOR_BGR2GRAY)


ctrs, hier = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0],reverse=True)
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = new[y:y+h, x:x+w]

    # show ROI
    immggg = cv2.resize(roi, (32, 32))
    image1 = cv2.copyMakeBorder(immggg, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
    print(immggg.shape)
    immgg = cv2.resize(image1, (32, 32))
    cv2.imwrite('/content/drive/MyDrive/images/letter.jpg',immgg)
    cv2_imshow(immgg)

    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.waitKey(0)
cv2_imshow(image)
cv2.waitKey(0)

