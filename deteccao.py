import cv2
import mediapipe as mp

# Definindo a largura e altura da imagem
WIDTH = 1920
HEIGHT = 1080

# Carregando os módulos de detecção de mãos e desenho
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

# Inicializando o módulo de detecção de mãos
maos = mp_maos.Hands()

# Inicializando a câmera com a largura e altura definidas
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Laço de repetição para capturar a imagem da câmera e exibir na tela
while True:
    sucesso, imagem = camera.read()

    # O módulo de detecção de mãos do MediaPipe espera uma imagem no formato RGB
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    # Processa a imagem e retorna o resultado:
    # MULTI_HAND_LANDMARKS: informa as coordenadas normalizadas dos pontos da mão
    # MULTI_AND_WORLD_LANDMARKS: informa as coordenadas em metros
    # MULTI_HANDEDNESS: informa o lado da mão (esquerda/direita) + probabilidade de acerto
    resultado = maos.process(imagem_rgb)

    # Desenha as marcações das mãos na imagem, caso existam
    if resultado.multi_hand_landmarks:
        for marcacao_maos in resultado.multi_hand_landmarks:
            mp_desenho.draw_landmarks(imagem, marcacao_maos, mp_maos.HAND_CONNECTIONS)

    if not sucesso:
        break

    cv2.imshow("Imagem", imagem)

    # Verifica se a tecla ESC (código 27) foi pressionada
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera a câmera e fecha todas as janelas
camera.release()
cv2.destroyAllWindows()
