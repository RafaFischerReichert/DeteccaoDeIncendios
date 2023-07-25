import cv2
import numpy as np


def segment_image(frame, method):
    if method == "motion":
        if "prev_gray" not in segment_image.__dict__:
            segment_image.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            segment_image.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        threshold = 10
        segmented_regions = magnitude > threshold
        # Convertendo para valores inteiros (0s e 1s)
        segmented_regions = segmented_regions.astype("int")

        segment_image.prev_gray = gray

    elif method == "color":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 120, 70])  # Valores mínimos de H, S e V
        upper_color = np.array([30, 255, 255])  # Valores máximos de H, S e V
        mask = cv2.inRange(hsv, lower_color, upper_color)
        segmented_regions = mask

    elif method == "flicker":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        threshold = 2  # Ajuste esse valor conforme necessário
        segmented_regions = std_dev > threshold

        # Convertendo para valores booleanos (True/False)
        segmented_regions = segmented_regions.astype(bool)

    else:
        raise ValueError(
            "Método de segmentacao inválido. Escolha entre 'motion', 'color' ou 'flicker'."
        )

    return segmented_regions


cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Segmentação por movimento
    motion_regions = segment_image(frame, "motion")

    # Segmentação por cor
    color_regions = segment_image(frame, "color")

    # Segmentação por cintilação (flicker)
    flicker_regions = segment_image(frame, "flicker")

    # Mostrar resultados em três janelas separadas
    cv2.imshow("Segmentacao por Movimento", motion_regions.astype("uint8") * 255)
    cv2.imshow("Segmentacao por Cor", color_regions.astype("uint8") * 255)
    cv2.imshow("Segmentacao por Cintilancia", flicker_regions.astype("uint8") * 255)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
