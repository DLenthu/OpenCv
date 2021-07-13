import cv2
import numpy as np

image_bgr = cv2.imread("template.jpg")
image_gray = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)

template = cv2.imread("matching.jpg",0)

w,h = template.shape[::-1]

res = cv2.matchTemplate(image_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.75

loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(image_bgr,pt,(pt[0]+w,pt[1]+h),(0,255,255),2)

cv2.imshow("detected",image_bgr)
cv2.waitKey(0)