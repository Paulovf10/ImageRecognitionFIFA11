import cv2


# Lista para armazenar os rastreadores e suas respectivas caixas delimitadoras
trackers = []
boxes = []

# Abra o vídeo
cap = cv2.VideoCapture('static/partida2.mp4')

# Leia o primeiro quadro
ret, frame = cap.read()

pause = False

while True:
    # Leia o próximo quadro
    if not pause:
        ret, frame = cap.read()

    for i, tracker in enumerate(trackers):
        # Atualize o rastreador com o novo quadro
        success, box = tracker.update(frame)

        # Desenhe um retângulo ao redor do jogador se o rastreamento foi bem-sucedido
        if success:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        else:
            cv2.putText(frame, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('MultiTracker', frame)

    # Se a tecla 'a' for pressionada, pause o vídeo e adicione um novo rastreador
    key = cv2.waitKey(1)
    if key == ord('a'):
        pause = True
        box = cv2.selectROI('MultiTracker', frame)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, box)
        trackers.append(tracker)
        boxes.append(box)
        cv2.destroyAllWindows()

    # Se a tecla 's' for pressionada, continue o vídeo
    if key == ord('s'):
        pause = False

    # Saia se a tecla 'esc' for pressionada
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
