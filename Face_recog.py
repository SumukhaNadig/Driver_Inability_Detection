import face_recognition
import pyrebase



if __name__ == '__main__':
    config = {
        "apiKey": "AIzaSyCtXef30sl_42IvHOCVR6LjeF4UvAdaBYU",
        "authDomain": "garage-opener-9ceb2.firebaseapp.com",
        "databaseURL": "https://garage-opener-9ceb2.firebaseio.com",
        "projectId": "garage-opener-9ceb2",
        "storageBucket": "garage-opener-9ceb2.appspot.com",
        "messagingSenderId": "256687415730"
       }
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    i=1
  #Download all registered driver images from Firebase storage
    """while 1:
        try:
            storage.child("FaceReg/"+str(i)+".jpg").download("drive/My Drive/Colab/NMIT"+str(i)+".jpg")
            i=i+1
        except:
            break
        storage.child("Images/test.jpg").download("drive/My Drive/Colab/NMIT/user.jpg")
    """
    driver_face=face_recognition.load_image_file("user.jpg")
    driver_face_encoding = face_recognition.face_encodings(driver_face)[0]
    j=1
    flag=0
    while j<=i:
        stored_face = face_recognition.load_image_file(str(j)+".jpg")
        stored_face_encoding = face_recognition.face_encodings(stored_face)[0]
        results = face_recognition.compare_faces([driver_face_encoding], stored_face_encoding)
        j=j+1
        if results[0]==True:
            flag=1
            print("Match")
            break
          #Code to extract blink frequency and blink duration from firbase database
          #Code to continuously record video and compared with retrieved personalized threshold
          #Alert code conditions
        if j>i and flag==0:
            print("No Match")
        #Code to take 1-minute video of new driver and calculate blink freq and blink duration, and insert into database
        #Code to continuously record video and compared with calculated personalized threshold
        #Alert code conditions