import cv2

img = cv2.imread("fog.jpg",0)

img1 = cv2.equalizeHist(img)

cv2.imshow("Orig",img)
cv2.imshow("Enhance",img1)
cv2.waitKey(0)