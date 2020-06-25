# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import numpy as np
import argparse
import time
import cv2
import os
import dlib

arr=list(range(68,81))+list(range(17,27))
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []
	landmarks = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")


			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			height = endY - startY
			width = endX - startX

			face = frame[startY:endY, startX:endX]
			landmarks.append(predictor(gray, dlib.rectangle(int(startX),int(startY),int(endX),int(endY))))
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX-int(0.01*width), startY-int(0.01*height), endX+int(0.01*width), endY+int(0.01*height)))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds,landmarks)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
#vs = VideoStream(src=0).start()
#time.sleep(2.0)

# loop over the frames from the video stream
while True:
	_, frame = cap.read()
	#frame = vs.read()
	#frame = imutils.resize(frame, width=400)
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds, landmarks) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred,landmark) in zip(locs, preds, landmarks):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(inc, mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		'''label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)'''
		if(mask>withoutMask and mask>inc):
			label = "Mask"
		elif(withoutMask>mask and withoutMask>inc):
			label = "No mask"
		else:
			label = "Mask worn incorrectly"
		if label=="Mask":
			color = (0,255,0)
		elif label=="No mask":
			color = (0,0,255)
		else:
			color = (0,165,255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(inc, mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		for n in arr:
			x = landmark.part(n).x
			y = landmark.part(n).y
			cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()