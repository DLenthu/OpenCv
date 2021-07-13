import cv2
import numpy as np

width,height = 500,700
img = cv2.imread("card.jpg")
pts1 = np.float32([[81,54],[435,120],[551,669],[167,743]])
pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
img_output = cv2.warpPerspective(img,matrix,(width,height))

for i in range(4):
    cv2.circle(img,(pts1[i][0],pts1[i][1]),5,(0,0,255),cv2.FILLED)

cv2.imshow("Original image",img)
cv2.imshow("Transformed image",img_output)
cv2.waitKey(0)