from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib as dl
import cv2
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np
import time
from firebase import firebase

def EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


def registerEAR():
    BlinkThresh = 0.23
    BlinkFrames = 3
    avgEAR = 0
    counter = 0
    count=0
    TOTAL = 0
    detector = dl.get_frontal_face_detector()
    predict = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vs = VideoStream(src=0).start()

    fileStream = False
    time.sleep(1.0)

    a = time.time() + 5

    while time.time() < a:        
        counter+=1


        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pixels = detector(gray, 0)

        for pixel in pixels:
            shape = predict(gray,pixel)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)

            ear = (leftEAR + rightEAR ) /2

            avgEAR += ear

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.imshow("Frame",frame)


            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
    return avgEAR/counter

def registerBlFreq(newEAR):
    BlinkThresh = newEAR
    BlinkFrames = 3

    count = 0

    total = 0

    detector = dl.get_frontal_face_detector()
    predict = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vs = VideoStream(src=0).start()
    fileStream = False
    time.sleep(1.0)

    a = time.time() + 10

    while time.time() < a:
        if fileStream and not vs.more():
            break

        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pixels = detector(gray, 0)

        for pixel in pixels:
            shape = predict(gray, pixel)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < BlinkThresh:
                count +=1
                # print(count)

            else:
                if count >= BlinkFrames:
                    total +=1

                count = 0

            cv2.imshow("Frame",frame)
            cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
    return total


def register():
    avg = registerEAR()
    blavg = registerBlFreq(avg)

    print(avg)
    print(blavg)
    fire = firebase.FirebaseApplication("https://garage-opener-9ceb2.firebaseio.com/")
  
    c=fire.get("/Drivers",None)
    c_keys=list(c.keys())
    m=max(c_keys)
    #print(m)
    nex=str(int(m)+1)
    result=fire.put("/Drivers",nex,'')
    result=fire.put("/Drivers/"+nex,'EAR',avg)
    result=fire.put("/Drivers/"+nex,'BlinkFreq',blavg)
    #print(result)   
    
    


register()



    

            








        






