from ultralytics import YOLO
import cv2
import math
import imutils
import os
import numpy as np

# start webcam
dataPath = "Data_CTS"  # Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Leyendo el modelo
face_recognizer.read("multiple.xml")
cap = cv2.VideoCapture("pruebas/Alex1_Cam1.mp4")
cap.set(3, 640)
cap.set(4, 480)
iou_thres = 0.7
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
model = YOLO("yolov8n-seg.pt")
net = cv2.dnn.readNet("yolov8n-seg.onnx")
classNames = model.names
count = 0
CUDA = True

while True:
    if CUDA == True:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    success, img = cap.read()
    # img=cv2.medianBlur(img,9)
    results = model(img, stream=True, iou=iou_thres)
    predict = model.predict(img)
    salida = (predict[0].masks.masks[0].numpy() * 255).astype("uint8")
    contours, hierarchy = cv2.findContours(
        salida, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # coordinates
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    auxFrame_textura = gray.copy()
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    auxFrame_color = gray_color.copy()
    auxFrame_color1 = img_color.copy()
    for r in results:
        boxes = r.boxes
        # auxFrame = contours.copy()
        for box in boxes:
            # bounding box
            cls = int(box.cls[0])
            if classNames[cls] != "person":
                continue
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            x, y, w, h = cv2.boundingRect(contours[0])
            textura = auxFrame_textura[y1:y2, x1:x2]
            color = auxFrame_color[y1:y2, x1:x2]
            color1 = auxFrame_color1[y1:y2, x1:x2]
            roi = salida[y : y + h, x : x + w]
            _, silhouette = cv2.threshold(roi, 1, 255, cv2.THRESH_BINARY)
            silhouette = cv2.resize(silhouette, (x2 - x1, y2 - y1))
            silueta = silhouette
            cv2.imshow("binario", silhouette)
            cv2.imshow("color", color1)
            result_silueta = face_recognizer.predict(silueta)
            result_textura = face_recognizer.predict(textura)
            result_color = face_recognizer.predict(color)
            # Obtener valores normalizados de confianza
            conf_silueta = 1 - result_silueta[1] / 100
            conf_textura = 1 - result_textura[1] / 100
            conf_color = 1 - result_color[1] / 100
            min_conf = min(conf_silueta, conf_textura, conf_color)
            if min_conf == conf_silueta:
                index = result_silueta[0]
            elif min_conf == conf_textura:
                index = result_textura[0]
            else:
                index = result_color[0]
            # Obtener la etiqueta de la imagen correspondiente
            etiqueta = imagePaths[index]
            print("resultado_silueta", result_silueta)
            print("resultado_textura", result_textura)
            print("resultado_color", result_color)
            results = [
                (result_silueta[0], result_silueta[1]),
                (result_textura[0], result_textura[1]),
                (result_color[0], result_color[1]),
            ]
            results = sorted(results, key=lambda x: x[1], reverse=True)
            print("resultado del convinado", results)
            if results[0][1] < 70:
                cv2.putText(
                    img,
                    "%: {:.2f}".format(100 - results[0][1])
                    + " "
                    + "{}".format(imagePaths[results[0][0]]),
                    (x1, y1 - 25),
                    1,
                    1.2,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                if imagePaths[results[0][0]].startswith("desconocido"):
                    print("Persona desconocida")
                else:
                    print("Persona conocida")

    frame = imutils.resize(img, width=1000)
    cv2.imshow("Webcam", frame)

    # Exit condition
    k = cv2.waitKey(1)
    if k == 27 or count >= 500:
        break

cap.release()
cv2.destroyAllWindows()
