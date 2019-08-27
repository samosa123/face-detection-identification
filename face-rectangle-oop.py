import cv2
import numpy as np
import pickle
from PIL import Image
import os

class face_rec:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('C:\\Users\Asus\\Desktop\\face detection\\haarcascade_frontalface_default.xml')
        #eye_cascade = cv2.CascadeClassifier('C:\\Users\\Asus\\Desktop\\face detection\\haarcascade_eye.xml')
        face1_cascade = cv2.CascadeClassifier('C:\\Users\\Asus\\Desktop\\face detection\\haarcascade_frontalface_alt2.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainner.yml")
        self.labels = {"person_name": 1}
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            self.labels = {v:k for k,v in og_labels.items()}
        self.cap = cv2.VideoCapture(0)
	
    def face_d(self):
        while True:
            ret, img = self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                #recognize?
                id_, conf = self.recognizer.predict(roi_gray)
                if conf >= 45 and conf <= 85:
                    print(id_)
                    print(self.labels[id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = self.labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    # eyes = eye_cascade.detectMultiScale(roi_gray)
                    #for (ex,ey,ew,eh) in eyes:
                    # cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)
            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break
        self.cap.release()
        cv2.destroyAllWindows()
        
        print (name)
        
t = face_rec()
t.face_d()