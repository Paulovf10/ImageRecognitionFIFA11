import cv2
import numpy as np

# Carrega o modelo YOLO
net = cv2.dnn.readNet('yolo/yolov4.weights', 'yolo/yolov4.cfg')

# Carrega os nomes das classes
with open('yolo/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Carrega o vídeo
cap = cv2.VideoCapture('static/partida2.mp4')

while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    height, width, channels = frame.shape

    # Para cada detecção
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Obtem as coordenadas e dimensões da caixa delimitadora
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Desenha a caixa delimitadora
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2),
                              (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
