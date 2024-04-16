import cv2
import mediapipe as mp

# Definindo a largura e altura da imagem
WIDTH = 1920
HEIGHT = 1080

# Inicializando a câmera com a largura e altura definidas
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Laço de repetição para capturar a imagem da câmera e exibir na tela
while True:
    sucesso, imagem = camera.read()
    if not sucesso:
        break

    cv2.imshow("Imagem", imagem)

    # Verifica se a tecla ESC (código 27) foi pressionada
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera a câmera e fecha todas as janelas
camera.release()
cv2.destroyAllWindows()
