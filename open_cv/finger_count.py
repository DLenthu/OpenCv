import	cv2 
import threading
import	numpy	as	np

cascade = cv2.CascadeClassifier('fingertip_cascade.xml')
cap	= cv2.VideoCapture(0) 
scaling_factor = 1
while	True:				
    ret,frame = cap.read()	
    frame =	cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)
    frame = cv2.flip(frame,2)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    try:
        rects=cascade.detectMultiScale(gray,1.2,20)
        n=0			
        for(x,y,w,h) in	rects:					
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            n+=1
            cv2.putText(frame, str(n), (x+np.uint8(w/2)-8,y+np.uint8(h/2)+5), cv2.FONT_HERSHEY_SIMPLEX,.8, (255,0,0), 2)
            cv2.circle(frame,(x+np.uint8(w/2),y+np.uint8(h/2)),2,(0,0,255),-1)

    except:
        pass

    cv2.putText(frame, str(n), (10,450), cv2.FONT_HERSHEY_SIMPLEX,5, (200,255,120), 2)
    cv2.imshow('Video Feed',frame)
    
    if	cv2.waitKey(1)==27:								
        break

cap.release() 
cv2.destroyAllWindows()