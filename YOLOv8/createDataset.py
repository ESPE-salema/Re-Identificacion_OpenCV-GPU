import os
import cv2
import numpy as np
import imutils

personName = 'Alex'
classId = '0'
trainPath = 'YOLOv8/train'
validPath = 'YOLOv8/valid'

patch = validPath

if not os.path.exists(trainPath):
    print('Carpeta creada: ', trainPath)
    os.makedirs(trainPath + '/images')
    os.makedirs(trainPath + '/labels')

if not os.path.exists(validPath):
    print('Carpeta creada: ', validPath)
    os.makedirs(validPath)
    os.makedirs(validPath + '/images')
    os.makedirs(validPath + '/labels')

cap = cv2.VideoCapture('Tests/' + personName + '2_Cam1.mp4')
net = cv2.dnn.readNetFromONNX('YOLOv8/yolov8n-face.onnx')


def capture(CUDA):
    count = 0

    if CUDA == True:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    while True:

        # Capturar un frame de la cámara
        (ret, frame) = cap.read()

        if not ret:
            break

        [height, width, _] = frame.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = frame
        scale = length / 640

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        net.setInput(blob)
        outputs = net.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)
             ) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]
                                        ), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]

            scale_factor = 640 / max(height, width)
            resized_frame = cv2.resize(
                frame, None, fx=scale_factor, fy=scale_factor)

            cv2.imwrite(patch + '/images' + '/face_' + personName +
                        '_{}.jpg'.format(count), resized_frame)

            draw_bounding_box(frame, round(box[0] * scale), round(box[1] * scale),
                              round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale), count)

            count = count + 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Redimensionar el frame a un tamaño específico

        # Guardar el frame redimensionado como una imagen
        # cv2.imwrite(patch + '/face_' + personName + '_{}.png'.format(count), resized_frame)

        frame = imutils.resize(frame, width=640)

        if CUDA == True:
            cv2.imshow("CAPTURA GPU", frame)
        else:
            cv2.imshow("CAPTURA CPU", frame)


def draw_bounding_box(img, x, y, w, h, count):
    [height, width, _] = img.shape
    archivo = open(patch + '/labels' + '/face_' + personName +
                           '_{}.txt'.format(count), "w")
    archivo.write(
        classId + " {:.6f}".format(((x + w) / 2)/width) + " {:.6f}".format(((y + h) / 2)/height) + " {:.6f}".format((w - x)/width) + " {:.6f}".format((h - y)/height))
    archivo.close()
    cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
    cv2.putText(img, personName, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


capture(CUDA=True)
