import numpy as np
import cv2

cap = cv2.VideoCapture('M6 Motorway Traffic.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2() # 
kernel = np.ones((5,5),np.uint8)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('vehicleCounter.avi',fourcc, 20.0, (640,480))

vehicleName=0
count = 0

while(1):
	ret, frame = cap.read()
	fgmask=fgbg.apply(frame)

	frame = cv2.resize(frame, (640, 480))
	fgmask = cv2.resize(fgmask, (640, 480))

	med=cv2.medianBlur(fgmask,15)
	erosion = cv2.erode(med,kernel,iterations = 1)
	dilation = cv2.dilate(erosion,kernel,iterations = 1)
	fgmask = cv2.bitwise_and(fgmask, dilation)

	_,contours,_ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	for c in contours:
		if cv2.contourArea(c) < 500:
		    continue
		(x, y, w, h) = cv2.boundingRect(c)
		if y > 310 and y < 314:
			count += 1
		roi = frame[y:y + h, x:x + w]
		cv2.imwrite('vehicle/{}.png'.format(vehicleName), roi)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
		vehicleName=vehicleName+1

	total_cars = "Total Vehicles : " + str(count)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame,total_cars,(0,470),font,1,(0,255,0),2,cv2.LINE_AA)
	cv2.line(frame, (0, 310), (640, 310), (255, 0, 255), 3)

	cv2.imshow('fgmask',fgmask)
	cv2.imshow('frame',frame)

	#out.write(frame)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()