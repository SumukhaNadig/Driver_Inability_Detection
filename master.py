from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib as dl
import cv2
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np
import time

def EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def MAR(mou):
	X   = distance.euclidean(mou[0], mou[6])
	Y1  = distance.euclidean(mou[2], mou[10])
	Y2  = distance.euclidean(mou[4], mou[8])
	Y   = (Y1+Y2)/2.0
	mar = Y/X
	return mar

def Drowsy_detection():
    BlinkThresh = 0.23
    BlinkFrames = 3
    
    MouthThresh = 0.75
    yawnStatus = False
    yawns = 0

    SleepThresh = 0.25
    SleepFrames = 20

    COUNTER = 0
    TOTAL = 0

    detector = dl.get_frontal_face_detector()
    predict = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


    vs = VideoStream(src=0).start()

 
    time.sleep(1.0)
    flag=0
 
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pixels = detector(gray, 0)
        
        prev_yawn_status = yawnStatus

        for pixel in pixels:
            
            shape = predict(gray, pixel)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            
            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)
            mouEAR = MAR(mouth)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)


            if ear < BlinkThresh:
                COUNTER += 1
                flag+=1
                print(flag)
                if flag >= SleepFrames:
                    cv2.putText(frame,"***** ALERT! *****",(10,250),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame," You seem to be sleepy! Please get some rest!", (10,300),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
						
            else:
                flag = 0
                if COUNTER >= BlinkFrames:
                    TOTAL += 1
                COUNTER = 0
                
            if mouEAR > MouthThresh:
                cv2.putText(frame, "Yawning ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawnStatus = True
                output_text = "Yawn Count: " + str(yawns + 1)
                cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
            else:
                yawnStatus = False

            if prev_yawn_status == True and yawnStatus == False:
                yawns+=1

            cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame",frame)
            
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
     
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
 
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
	
Drowsy_detection()