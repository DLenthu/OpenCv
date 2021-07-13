import cv2
import numpy as np


img = cv2.imread("fractal.jpg",cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift)) 
print(magnitude_spectrum)

magnitude_spectrum = np.asarray(magnitude_spectrum,dtype = np.uint8)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum)

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows() 