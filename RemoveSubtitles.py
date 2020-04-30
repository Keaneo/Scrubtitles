#Imports
import cv2
import os
import numpy as np
import pytesseract as pt
from pytesseract import Output

#Read single image
#Change this to the path of the image with subtitles
img = cv2.imread('image.jpg')

#Main Process
#Detect subtitles, create mask, inpaint that area
mask = np.zeros(img.shape, np.uint8)
recogImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
recogImg = cv2.threshold(recogImg, 240, 255, cv2.THRESH_BINARY)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,5))
recogImg = cv2.morphologyEx(recogImg, cv2.MORPH_CLOSE, kernel)
recogImg = cv2.dilate(recogImg, kernel, iterations=3)
contours, hierarchy = cv2.findContours(recogImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask[y:y+h, x:x+w] = recogImg[y:y+h, x:x+w]
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.GaussianBlur(mask, (3,3), 0)

cleanedImg = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
 
#Show all stages of process
cv2.imshow("original", img)
cv2.imshow("b&w", recogImg)
cv2.imshow("mask", mask)
cv2.imshow("clean", cleanedImg)

#Press key to close
cv2.waitKey(0)
