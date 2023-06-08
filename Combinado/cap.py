from ultralytics import YOLO
import cv2
import math 
import imutils
import numpy as np
import os
from multiprocessing import Process
# start webcam
video="pruebas/Alex1_Cam1.mp4"
name="Alex"
def Color():
    personName = name
    dataPath = 'Data_CTS' 
    personPath = dataPath + '/' + personName

    if not os.path.exists(personPath):
        print('Carpeta creada: ',personPath)
        os.makedirs(personPath)
    cap = cv2.VideoCapture(video)
    cap.set(3, 640)
    cap.set(4, 480)
    iou_thres = 0.7
    count = 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # model
    model = YOLO("yolov8n.pt")
    # object classes
    classNames = model.names
    while True:
        success, img = cap.read()
        #img=cv2.medianBlur(img,9)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        results = model(img, stream=True,iou=iou_thres)
        # coordinates
        for r in results:
            boxes = r.boxes
            auxFrame = img.copy()
            for box in boxes:
                # bounding box
                cls = int(box.cls[0])
                if classNames[cls]!="person":
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                rostro = auxFrame[y1:y2,x1:x2]
                cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
                count = count + 1
        frame =  imutils.resize(img, width=1000)
        cv2.imshow('Webcam', frame)
        k=cv2.waitKey(1)
        if k==27 or count>=300:
            break
    cap.release()
    cv2.destroyAllWindows()

def Textura():
    personName = name
    dataPath = 'Data_CTS' 
    personPath = dataPath + '/' + personName

    if not os.path.exists(personPath):
        print('Carpeta creada: ',personPath)
        os.makedirs(personPath)
    cap = cv2.VideoCapture(video)
    cap.set(3, 640)
    cap.set(4, 480)
    iou_thres = 0.7
    count = 300
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # model
    model = YOLO("yolov8n.pt")
    # object classes
    classNames = model.names
    while True:
        success, img = cap.read()
        #img=cv2.medianBlur(img,9)
        results = model(img, stream=True,iou=iou_thres)
        # coordinates
        for r in results:
            boxes = r.boxes
            auxFrame = img.copy()
            for box in boxes:
                # bounding box
                cls = int(box.cls[0])
                if classNames[cls]!="person":
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                rostro = auxFrame[y1:y2,x1:x2]
                cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
                count = count + 1
        frame =  imutils.resize(img, width=1000)
        cv2.imshow('Webcam', frame)
        k=cv2.waitKey(1)
        if k==27 or count>=600:
            break
    cap.release()
    cv2.destroyAllWindows()
    
def Silueta():
    personName = name
    dataPath = 'Data_CTS' 
    personPath = dataPath + '/' + personName

    if not os.path.exists(personPath):
        print('Carpeta creada: ',personPath)
        os.makedirs(personPath)
    cap = cv2.VideoCapture(video)
    cap.set(3, 640)
    cap.set(4, 480)
    iou_thres = 0.7
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    model = YOLO("yolov8n-seg.pt")
    classNames = model.names
    count = 600
    while True:
        success, img = cap.read()
        #img=cv2.medianBlur(img,9)
        results = model(img, stream=True,iou=iou_thres)
        predict=model.predict(img)
        salida=(predict[0].masks.masks[0].numpy()*255).astype("uint8")
        #salida = cv2.Canny(salida,100,200)
        contours, hierarchy = cv2.findContours(salida, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # coordinates
        for r in results:
            boxes = r.boxes
            #auxFrame = contours.copy()
            for box in boxes:
                # bounding box
                cls = int(box.cls[0])
                if classNames[cls]!="person":
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                # Extract ROI from binary mask
                x, y, w, h = cv2.boundingRect(contours[0])
                roi = salida[y:y+h, x:x+w]
                # Apply thresholding to obtain silhouetteq
                _, silhouette = cv2.threshold(roi, 1, 255, cv2.THRESH_BINARY)
                # Resize silhouette to match original image size
                silhouette = cv2.resize(silhouette, (x2-x1, y2-y1))
                # Blend silhouette with original image
                img[y1:y2, x1:x2] = cv2.bitwise_and(img[y1:y2, x1:x2], img[y1:y2, x1:x2], mask=cv2.bitwise_not(silhouette))
                # Recognize face in silhouette
                rostro = silhouette
                cv2.imshow('Webcamsss', silhouette)
                cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
                count += 1
        # Display frames
        frame =  imutils.resize(img, width=1000)
        cv2.imshow('Webcam', frame)
        frame1 =  imutils.resize(salida, width=1000)
        cv2.imshow("binario",frame1)
        
        # Exit condition
        k=cv2.waitKey(1)
        if k==27 or count>=900:
            break
            
    cap.release()
    cv2.destroyAllWindows()

def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    print("PROCESADORES",p)
    p.start()
    proc.append(p)
  for p in proc:
    print("SALIDA PROCESADORES",p)
    p.join()
if __name__ == '__main__':
  runInParallel(Textura,Color,Silueta)