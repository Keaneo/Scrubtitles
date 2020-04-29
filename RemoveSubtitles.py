import cv2
import os
import numpy as np
import pytesseract as pt
from pytesseract import Output

#img = cv2.imread('test.jpg')
img = cv2.imread('E:/Media/Videos/Test/frame10.jpg')

mask = np.zeros(img.shape, np.uint8)
recogImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#recogImg = cv2.GaussianBlur(recogImg, (3,3), 0)
#recogImg = cv2.medianBlur(recogImg, 9)
recogImg = cv2.threshold(recogImg, 240, 255, cv2.THRESH_BINARY)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
recogImg = cv2.morphologyEx(recogImg, cv2.MORPH_CLOSE, kernel)
#recogImg = cv2.morphologyEx(recogImg, cv2.MORPH_OPEN, kernel)
recogImg = cv2.dilate(recogImg, kernel, iterations=3)
#recogImg = 255 - recogImg
#subtitlesString = pt.image_to_string(recogImg, lang='eng', config='--psm 3')
#subtitles = pt.image_to_data(recogImg, output_type=Output.DICT)
#n_boxes = len(subtitles['level'])
contours, hierarchy = cv2.findContours(recogImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask[y:y+h, x:x+w] = recogImg[y:y+h, x:x+w]
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.GaussianBlur(mask, (3,3), 0)


# for i in range(n_boxes):
#     if i > 0:
#         (x,y,w,h) = (subtitles['left'][i], subtitles['top'][i], subtitles['width'][i], subtitles['height'][i])
#         mask = cv2.rectangle(mask, (x,y), (x+w, y+h), (255,255,255), -1)
#         print(x, y)
#         print(w, h)
#print(subtitlesString)
#mask = recogImg
#
cleanedImg = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
 
cv2.imshow("original",img)
cv2.imshow("b&w", recogImg) 
cv2.imshow("mask",mask)
cv2.imshow("clean", cleanedImg)

cv2.waitKey(0)
