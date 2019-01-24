from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib as dl
import cv2
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np
import time
import pyrebase
import face_recognition
import cv2
from firebase import firebase
import time
import playsound
import threaded
@threaded.Threaded

def sound_alarm(path='alarm.wav'):
	# play an alarm sound
	playsound.playsound(path)

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
    return 0.85*(avgEAR/counter)

def registerBlFreq(newAvg):
    BlinkThresh = newAvg
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
  
    c = fire.get("/Drivers",None)
   
    print(c)
    c_keys=list(c.keys())
    if len(c_keys)==0:
        nex='1'
    else:
        m=max(c_keys)
        #print(m)
        nex=str(int(m)+1)
    # result=fire.put("/Drivers','')
    result=fire.put("/Drivers",nex,'')
    result=fire.put("/Drivers/"+nex,'EAR',avg)
    result=fire.put("/Drivers/"+nex,'BlinkFreq',blavg)
    #print(result)   
    
    config = {
    "apiKey": "AIzaSyCtXef30sl_42IvHOCVR6LjeF4UvAdaBYU",
    "authDomain": "garage-opener-9ceb2.firebaseapp.com",
    "databaseURL": "https://garage-opener-9ceb2.firebaseio.com",
    "projectId": "garage-opener-9ceb2",
    "storageBucket": "garage-opener-9ceb2.appspot.com",
    "messagingSenderId": "256687415730",
    "serviceAccount": "garage-opener-9ceb2-firebase-adminsdk-bh3um-c3f4bb9f52.json",
    }
    firebase1 = pyrebase.initialize_app(config)
    storage = firebase1.storage()
    storage.child("FaceReg/"+nex+".jpg").put("Images/user.jpg")

def Drowsy_detection(ear,freq):
    alarm_on = False
    BlinkThresh = ear
    BlinkFrames = 3
    FreqThresh=freq

    MouthThresh = 0.60
    yawnStatus = False
    yawns = 0

    SleepThresh = 0.25
    SleepFrames = 18

    COUNTER = 0
    TOTAL = 0

    detector = dl.get_frontal_face_detector()
    predict = dl.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


    vs = VideoStream(src=0).start()

    # fl=0
    time.sleep(1.0)
    flag=0
    t1=time.time()
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
                #print(flag)
                if flag >= SleepFrames:
                    if not alarm_on:
                        alarm_on = True

                    thread = sound_alarm()
                    thread.start()
                    thread.join()
                    # fl=1
                    cv2.putText(frame,"***** ALERT! *****",(10,250),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame," You are incapable of driving! Please get some rest!", (10,300),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2)
						
            else:
                flag = 0
                alarm_on = False
                if COUNTER >= BlinkFrames:
                    TOTAL += 1
                COUNTER = 0
                baseURL='https://api.thingspeak.com/update?api_key=UP2MYEJSHUGSTV7D&field1='
                c=1
                """f = urllib.request.urlopen(baseURL + str(c))
                f.read()
                f.close()"""
                
            if mouEAR > MouthThresh:
                cv2.putText(frame, "Yawning ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawnStatus = True
                output_text = "Yawn Count: " + str(yawns)
                cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
            else:
                yawnStatus = False
                

            if prev_yawn_status == True and yawnStatus == False:
                yawns+=1
                output_text = "Yawn Count: " + str(yawns)
                cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)

            cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame",frame)
            
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            """t2=time.time()
            if t2-t1>=5:
                a = time.time()+4
                while time.time()<=a:
                    cv2.putText(frame, "Checking measured driver data:", (10,10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if yawns>0:
                        cv2.putText(frame, "You yawned {:.2f} time(s). You might be sleepy".format(yawns), (30,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        yawns=0
                    if TOTAL>1.2*FreqThresh*2:
                        cv2.putText(frame, "You seem to be mildly tired. Beware!", (30,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif TOTAL>1.5*FreqThresh*2:
                        cv2.putText(frame, "You seem to be extremely tired. Please take some rest!", (30,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                    cv2.imshow("Frame", frame)  

                TOTAL=0
                t1=t2
                t2=time.time()
                """
        cv2.imshow("Frame", frame)
        """if fl==1:
            baseURL='https://api.thingspeak.com/update?api_key=UP2MYEJSHUGSTV7D&field1='
            c=0
            f = urllib.request.urlopen(baseURL + str(c))
            f.read()
            f.close()
            fl=0"""
        key = cv2.waitKey(1) & 0xFF
    
 
        if key == ord("q"):
            break
        

    cv2.destroyAllWindows()
    vs.stop()
	
def main():
    config = {
        "apiKey": "AIzaSyCtXef30sl_42IvHOCVR6LjeF4UvAdaBYU",
        "authDomain": "garage-opener-9ceb2.firebaseapp.com",
        "databaseURL": "https://garage-opener-9ceb2.firebaseio.com",
        "projectId": "garage-opener-9ceb2",
        "storageBucket": "garage-opener-9ceb2.appspot.com",
        "messagingSenderId": "256687415730",
        "serviceAccount": "garage-opener-9ceb2-firebase-adminsdk-bh3um-c3f4bb9f52.json",
       }
    fire = pyrebase.initialize_app(config)
    storage = fire.storage()
    i=63
  #Download all registered driver images from Firebase storage
    while 1:
        try:
            storage.child("FaceReg/"+str(i)+".jpg").download("Images/"+str(i)+".jpg")
            i=i+1
        except:
            break
        # storage.child("Images/test.jpg").download("Desktop/Images/user.jpg")
    
    camera = cv2.VideoCapture(0)
    val,image=camera.read()
    cv2.imwrite("Images/user.jpg",image)
    driver_face=face_recognition.load_image_file("Images/user.jpg")
    driver_face_encoding = face_recognition.face_encodings(driver_face)[0]
    j=63
    flag=0
    while j<i:
        stored_face = face_recognition.load_image_file("Images/"+str(j)+".jpg")
        stored_face_encoding = face_recognition.face_encodings(stored_face)[0]
        results = face_recognition.compare_faces([driver_face_encoding], stored_face_encoding)
        if results[0]==True:
            flag=1
            print("Face Recognition Match Found") 
            fire2 = firebase.FirebaseApplication("https://garage-opener-9ceb2.firebaseio.com/")
            res=fire2.get("/Drivers/"+str(j),None)
            print(res)
            driver_ear=res["EAR"]
            driver_blinkf=res["BlinkFreq"]
            Drowsy_detection(driver_ear,driver_blinkf)
            break
        j+=1
    if j>=i and flag==0:
        print("No Match Found. Please register first")  
        register()

main()