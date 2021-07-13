import cv2

import numpy as np

cap = cv2.VideoCapture(0)


while True:
    _,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur_img = cv2.medianBlur(gray,9)

    edges = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,5)

    color = cv2.bilateralFilter(img,9,300,300)

    cartoon = cv2.bitwise_and(color,color,mask = edges)

    cv2.imshow("Cartoon",cartoon)
    cv2.waitKey(1)
cv2.destroyAllWindows()