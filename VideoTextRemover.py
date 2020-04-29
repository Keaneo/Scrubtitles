import cv2
import os
import numpy as np
import pytesseract as pt
from pytesseract import Output
import re 
import time

start_time = time.time()

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

#img = cv2.imread('test.jpg')
#dir = 'E:/Media/Videos/'
dir = 'C:/Users/liamk/ScrubtitlesFiles/'
vid = cv2.VideoCapture(os.path.join(dir, 'test.mp4'))
frame_counter = 0

if not os.path.exists(os.path.join(dir,'Test')):
    os.mkdir(os.path.join(dir,'Test'))
    print("Directory " , os.path.join(dir,'Test') ,  " Created ")
else:    
    print("Directory " , os.path.join(dir,'Test') ,  " already exists")

os.chdir(os.path.join(dir,'Test'))

while (frame_counter < vid.get(cv2.CAP_PROP_FRAME_COUNT)):
    ret, img = vid.read()
    name = "frame%d.jpg" % (frame_counter)
    if not ret:
        break
    if not os.path.exists(os.path.join(dir,"Test/frame%d.jpg" % (frame_counter))):
        mask = np.zeros(img.shape, np.uint8)
        recogImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #recogImg = cv2.threshold(recogImg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        recogImg = cv2.threshold(recogImg, 200, 255, cv2.THRESH_BINARY)[1]
        #recogImg = cv2.GaussianBlur(recogImg, (3,3), 0)
        #recogImg = cv2.medianBlur(recogImg, 9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        recogImg = cv2.morphologyEx(recogImg, cv2.MORPH_CLOSE, kernel)
        recogImg = cv2.dilate(recogImg, kernel, iterations=3)
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #recogImg = cv2.morphologyEx(recogImg, cv2.MORPH_OPEN, kernel, iterations=1)
        #recogImg = 255 - recogImg
        subtitlesString = pt.image_to_string(recogImg, lang='eng', config='--psm 7')
        subtitles = pt.image_to_data(recogImg, output_type=Output.DICT)
        n_boxes = len(subtitles['level'])
        for i in range(n_boxes):
            if i > 0:
                (x,y,w,h) = (subtitles['left'][i], subtitles['top'][i], subtitles['width'][i], subtitles['height'][i])
                if not x == 0 or not y == 0:
                    mask = cv2.rectangle(mask, (x,y), (x+w, y+h), (255,255,255), -1)
                    print(x, y)
                    print(w, h)
        mask = recogImg
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cleanedImg = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        #cv2.imshow("original",img)
        #cv2.imshow("clean", cleanedImg)        
        cv2.imwrite(name, cleanedImg)
        frame_counter += 1
        #cv2.waitKey(0)

video = cv2.VideoWriter('output.avi', 0, int(vid.get(cv2.CAP_PROP_FPS)), (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
images = [img for img in os.listdir(os.path.join(dir,'Test')) if img.endswith(".jpg")]
images = sorted_nicely(images)
for image in images:
    print(image)
    video.write(cv2.imread(image))

video.release()
vid.release()
print("--- %s seconds ---" % (time.time() - start_time))

#print(subtitlesString) 

#cv2.imshow("b&w", recogImg) 
#cv2.imshow("mask",mask)



#mask = Text shapes from Tesseract ocr