import cv2

img = cv2.imread("card.jpg")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

print(len(contours))

cv2.drawContours(img,contours,-1,(0,255,0),3)

cv2.imshow("Image",img)
# cv2.imshow("Image_gray",imgray)
cv2.waitKey(0)
cv2.destroyAllWindows()