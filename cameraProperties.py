# Import required modules 
import cv2 
import numpy as np 
import os 
import glob 
import ffmpeg

frameWidth = 960	
frameHeight = 540
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

font = cv2.FONT_HERSHEY_DUPLEX
  
CHECKERBOARD = (7, 6) 
  
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
  

def getCameraMatrices():
	# Vector for 3D points 
	threedpoints = [] 
	  
	# Vector for 2D points 
	twodpoints = [] 
	  
	  
	#  3D points real world coordinates 
	objectp3d = np.zeros((1, CHECKERBOARD[0]  
		              * CHECKERBOARD[1],  
		              3), np.float32) 
	objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
		                       0:CHECKERBOARD[1]].T.reshape(-1, 2) 
	prev_img_shape = None
	  
	images = glob.glob('*.jpg') 
	  
	for filename in images: 
		image = cv2.imread(filename) 
		grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

		ret, corners = cv2.findChessboardCorners( 
		            grayColor, CHECKERBOARD,  
		            cv2.CALIB_CB_ADAPTIVE_THRESH  
		            + cv2.CALIB_CB_FAST_CHECK + 
		            cv2.CALIB_CB_NORMALIZE_IMAGE) 
	  
	# them on the images of checker board 
		if ret == True: 
			threedpoints.append(objectp3d) 
	  
			# Refining pixel coordinates 
			# for given 2d points. 
			corners2 = cv2.cornerSubPix( 
		    		grayColor, corners, (11, 11), (-1, -1), criteria) 
	  
			twodpoints.append(corners2) 
	  
			# Draw and display the corners 
			image = cv2.drawChessboardCorners(image,  
		                                  CHECKERBOARD,  
		                                  corners2, ret) 
	  
			cv2.imshow('img', image) 
			cv2.waitKey(10) 
	  
	cv2.destroyAllWindows() 
	  
	h, w = image.shape[:2] 
	  
	  
	# Perform camera calibration by 
	ret, matrix, distortion, rVecs, tVecs = cv2.calibrateCamera( 
	    threedpoints, twodpoints, grayColor.shape[::-1], None, None) 
	  
	  
	# Displayig required output 
	print(" Camera matrix:") 
	print(matrix) 
	  
	print("\n Distortion coefficient:") 
	print(distortion) 
	  
	print("\n Rotation Vectors:") 
	print(rVecs) 
	  
	print("\n Translation Vectors:") 
	print(tVecs) 


	newcameramatrix, roi=cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h))
	
	return matrix, distortion, rVecs, tVecs, newcameramatrix

def empty(a):
	pass

def draw(img, corners, imgpts):
	corner = tuple(corners[0].ravel())
	img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
	img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
	img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
	return img


def check_rotation(video_file_path):
	meta_data = ffmpeg.probe(video_file_path)

	rotate_code = None
	try:
		rotate_tag = int(meta_data['streams'][0]['tags']['rotate'])
	except KeyError:
		rotate_tag = 0

	if rotate_tag == 90:
		rotate_code = cv2.ROTATE_90_CLOCKWISE
	elif rotate_tag == 180:
		rotate_code = cv2.ROTATE_180
	elif rotate_tag == 270:
		rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

	return rotate_code


def correct_rotation(video_frame, rotate_code):
	if rotate_code is None:
		return video_frame
	else:
		return cv2.rotate(video_frame, rotate_code)


def brightnessAndContrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def preProcess(img):
	# CLAHE (Contrast Limited Adaptive Histogram Equalization)
	clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
	l, a, b = cv2.split(lab)  # split on 3 different channels

	l2 = clahe.apply(l)  # apply CLAHE to the L-channel

	lab = cv2.merge((l2,a,b))  # merge channels
	img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

	contrast = cv2.getTrackbarPos("Contrast", "Parameters")
	img = brightnessAndContrast(img, 0, contrast)

	imgBlur = cv2.GaussianBlur(img, (7,7), 7)
	imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)


	black = cv2.getTrackbarPos("Black", "Parameters")
	white = cv2.getTrackbarPos("White", "Parameters")
	ret, imgGray = cv2.threshold(imgGray,black,white,0)

	threshold1 = cv2.getTrackbarPos("Threshold 1", "Parameters")
	threshold2 = cv2.getTrackbarPos("Threshold 2", "Parameters")
	imgCanny = cv2.Canny(imgGray, threshold1, threshold2)


	kernel = np.ones((5, 5))
	imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

	return imgDil


def getContours(img, imgContour):
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	for cnt in contours:
		area = cv2.contourArea(cnt)
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)




		if len(approx) == 4 and cv2.contourArea(cnt) > cv2.getTrackbarPos("Area", "Parameters"):
			cv2.drawContours(imgContour, contours, -1,(255, 0, 255), 7)

			x, y, w, h = cv2.boundingRect(approx)
			cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

			##cv2.putText(imgContour, "En fyrkant!", (x, y), font, 1, (0, 0, 0))
			return x, y
	return 0, 0

def calculateXYZ(u,v):

	X_center=10.2
	Y_center=9.1
	Z_center=91.1
	worldPoints=np.array([[X_center,Y_center,Z_center],
		[11.0,5.0,91.0],
		[20.8,5.0,91.4],
		[11.0,13.5,91.4],
		[20.8,13.5,91.9]], dtype=np.float32)


	imagePoints=np.array([[307,227],
		[315,187],
		[425,187],
		[314,282],
		[426,282]], dtype=np.float32)


	ret, rvec1, tvec1=cv2.solvePnP(worldPoints,imagePoints,matrix,distortion)
	rotationMatrix, jac=cv2.Rodrigues(rvec1)
	##extrinsicMatrix = np.column_stack((rotationMatrix,tvec1)) Useless

	scalingfactor = 162
	
	uv=np.array([[u,v,1]], dtype=np.float32)
	uv=uv.T
	suv=scalingfactor*uv
	xyzC=np.linalg.inv(matrix).dot(suv)
	xyzC=xyzC-tvec1
	XYZ=np.linalg.inv(rotationMatrix).dot(xyzC)
			
	return XYZ


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Black", "Parameters", 170, 255, empty)
cv2.createTrackbar("White", "Parameters", 255, 255, empty)
cv2.createTrackbar("Contrast", "Parameters", 50, 255, empty)
cv2.createTrackbar("Threshold 1", "Parameters", 150, 155, empty)
cv2.createTrackbar("Threshold 2", "Parameters", 255, 155, empty)
cv2.createTrackbar("Area", "Parameters", 10000, 50000, empty)

matrix, distortion, rVecs, tVecs, newcameramatrix = getCameraMatrices()

while(cap.isOpened()):
	success, img = cap.read()
	if success:

		imgContour = img.copy()
		undistort = img.copy()

		imgPrP = preProcess(img.copy())
		xCorner, yCorner = getContours(imgPrP, imgContour)
		
		dst = cv2.undistort(undistort, matrix, distortion, None, newcameramatrix)
		
		# crop the imagenewcameramatrix
		#x, y, w, h = roi
		#dst = dst[y:y+h, x:x+w]


		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[0],CHECKERBOARD[1]),None)
		if ret == True:
			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			# Find the rotation and translation vectors.
			ret,rVecs, tVecs = cv2.solvePnP(objp, corners2, matrix, distortion)
			# project 3D points to image plane
			imgpts, jac = cv2.projectPoints(axis, rVecs, tVecs, matrix, distortion)
			imgContour = draw(imgContour,corners2,imgpts)

			np.linalg.inv(matrix)
			
			rotationMatrix, jac=cv2.Rodrigues(rVecs)

			rotationMatrix=np.column_stack((rotationMatrix,tVecs))
			
			#(corners[1] * np.linalg.inv(matrix) - tvecs ) * np.linalg.inv(R_mtx)
			print(f"A inverse: {np.linalg.inv(matrix)}")
			print(f"Corner: {corners[1]}")
			XYZ = calculateXYZ(corners[0][0][0], corners[0][0][1])
			
			print(f"XYZ vec: {XYZ}")
			cv2.imshow('Axis',imgContour)

			
		#cv2.imshow("Source", img)
		cv2.imshow("Undistorted", dst)
		#cv2.imshow("Contours", imgContour)
		#cv2.imshow("Gray", imgGray)

	else:
		print('no video')
		cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
