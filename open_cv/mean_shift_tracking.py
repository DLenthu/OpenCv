import numpy as np
import cv2

cap = cv2.VideoCapture("vtest.avi")

ret,frame = cap.read()

x,y,w,h = 246,218,40,90
# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

track_window = (x,y,w,h)

roi = frame[y : y+h,x : x+w] 
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi,np.array((0. ,60. ,32.)),np.array((180. ,255., 255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)


term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT ,10, 1 )

cv2.imshow("First frame",roi)

while(True):
    re,frame = cap.read()

    # if ret == True:
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv_frame],[0],roi_hist,[0,180],1)
    ret,track_window = cv2.meanShift(dst,track_window,term_crit)
    x,y,w,h = track_window
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow("frame",dst)
    cv2.imshow("frame",frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # else:
    #     break


