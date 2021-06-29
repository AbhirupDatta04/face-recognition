# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 22:24:21 2021

@author: abhir
"""
import cv2

face_casc=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_casc=cv2.CascadeClassifier('haarcascade_eye.xml')



def detect(gray,frame):
    faces=face_casc.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)