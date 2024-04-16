import cv2
import mediapipe as mp

WIDTH = 1920
HEIGHT = 1080

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

while True:
    sucesso, imagem = camera.read()
    if not sucesso:
        break

    cv2.imshow("Imagem", imagem)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
