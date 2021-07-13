import easyocr
import cv2

img = cv2.imread("text.jpg")
reader = easyocr.Reader(['en'], gpu = False)
out = reader.readtext('text.jpg')

for ele in out:
    coords,text = ele[0],ele[1]
    cv2.rectangle(img,tuple(coords[0]),tuple(coords[2]),(0,255,0),2)
    cv2.putText(img,text,(coords[0][0],coords[0][1]-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

cv2.imshow("OCR Output",img)
cv2.waitKey(0)