#Python Script To detect skin color from camera
import numpy as np
import cv2

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):

	dim = None
	(h, w) = image.shape[:2]
    
	if width is None and height is None:
		return image

	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)

	else:
		r = width / float(w)
		dim = (width, int(h * r))

	resized = cv2.resize(image, dim, interpolation = inter)
	return resized


#Colour Range for Skin
lower = np.array([0,48,80], dtype='uint8')
upper = np.array([20,255,255], dtype='uint8')

camera = cv2.VideoCapture(0)

while True:
	(grabbed, frame) = camera.read()
	frame=resize(frame,width=400)
	converted = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(converted,lower,upper)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	mask = cv2.erode(mask, kernel, iterations = 2)
	mask = cv2.dilate(mask, kernel, iterations = 2)

	mask = cv2.GaussianBlur(mask, (3,3) , 0)
	skin = cv2.bitwise_and(frame,frame,mask=mask)
	cv2.imshow("images",np.hstack([frame,skin]))
	if cv2.waitKey(1) & 0xFF ==ord("q"):
		break

camera.release()
cv2.destroyAllWindows()

