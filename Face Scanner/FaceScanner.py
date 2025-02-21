import cv2
import numpy as np

# Cargar el modelo de Deep Learning
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Iniciar la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener dimensiones del frame
    h, w = frame.shape[:2]

    # Preprocesar el frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Detección de rostros
    net.setInput(blob)
    detections = net.forward()

    # Dibujar los rostros detectados
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Nivel de confianza de la detección

        if confidence > 0.5:
            # Obtener las coordenadas del rectángulo
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Dibujar el rectángulo
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            text = f"{confidence * 100:.2f}%"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow('Face Detection', frame)

    # Salir con 'q' o si la ventana se cierra
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Liberar recursos y cerrar ventanas correctamente
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # Asegura que OpenCV cierra completamente
