# Driver_Inability_Detection
This project aims to reduce the number of accidents that occur because of drivers dozing off and losing control over the vehicle using real time image processing.

Files:
1. Face_recog.py :
    The file is used to identify a driver by processing his image and comparing it with the files on the database and correspondingly download associated variables to set threshold for blink frequency and duration.
    
2. master.py :
    This file is the core of the project and measures the number of times the driver blinks, yawns and the amount of time his eyes remain closed to identify it as him sleeping and then raise an alert message. 
    This is basically done with the help of an opensource dataset which maps points on the image to facial features and then finding the ratio of height of the space between each eye to twice its width as this ratio goes lower and lower as the person closes his eyes.
    
 Function descriptions:
 EAR() : Calculates the ratio of vertical space between eyes to twice the width.
 MAR() : Calulates the same ratio for mouth
 Drowsy_detection() : Function which loops over each frame in the videostream and then segments the image to detect faces and convert it to                       grayscale to obtain the minimum convex set over the points to obtain a contour plot and blinks, yawns and sleep time
                      is calculated.
    
    
Dependencies:
scipy
imutils
dlib
cv2
numpy
time
pyrebase
face_recognition
firebase
