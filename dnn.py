import cv2
import imutils
import os
import time
import numpy as np
from numba import jit

#print('imagePaths=', imagePaths)
# ----------- READ DNN MODEL -----------
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Leyendo el modelo
# face_recognizer.read('modeloDnnPERSONAS.xml')

# Weights
# weights = "model/yolov4-tiny.weights"
# Configuration
# cfg = "model/yolov4-tiny.cfg"

# Class labels
class_name = []
with open("YOLOv8/classe.txt", "r") as f:
    class_name = [line.strip() for line in f.readlines()]

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.6

cap = cv2.VideoCapture("Tests/Alex2_Cam1.mp4")
# Load the model
net = cv2.dnn.readNet('YOLOv8/persons.onnx')

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{class_name[class_id]} ({confidence:.2f})'
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 0, 255), 3)
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
def Salida(CUDA):

    init = time.time()

    if CUDA == True:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        

    while True:

        (grabbed, frame) = cap.read()

        if not grabbed:
            break

        start = time.time()

        [height, width, _] = frame.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = frame
        scale = length / 640

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1 / 255, size=(640, 640))
        net.setInput(blob)
        outputs = net.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)
             ) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.1:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]
                                        ), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.7)

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': class_name[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections.append(detection)
            draw_bounding_box(frame, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                              round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

        if cv2.waitKey(1) & 0xFF == 27:
            break

        end = time.time()
        frame = imutils.resize(frame, width=640)
        fps = "FPS: %.2f " % (1 / (end - start))
        cv2.putText(frame, fps, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if CUDA == True:
            cv2.imshow("CAPTURA GPU", frame)
        else:
            cv2.imshow("CAPTURA CPU", frame)

    fin = time.time()
    final = fin - init
    #print("Tiempo de reproducción: {:.2f}".format(fps.elapsed()))
    #print("FPS aproximado: {:.2f}".format(fps.fps()))
    print("Tiempo de ejecución: {:.2f}".format(final))
    #print("Verdaderos Positivos: ", VP)
    #print("Verdaderos Negativos: ", VN)
    #print("Falsos Positivos: ", FP)
    #print("Falsos Negativos: ", FN)

Salida(CUDA=True)