import cv2
import mediapipe as mp
import numpy as np

# Definindo a largura e altura da imagem
WIDTH = 640
HEIGHT = 480

# Carregando os módulos de detecção de mãos e desenho
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

# Inicializando o módulo de detecção de mãos
maos = mp_maos.Hands()

# Inicializando a câmera com a largura e altura definidas
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


def encontra_coordenadas_maos(
    img: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, list[tuple[int, int, int]]]]]:
    """
    Função que encontra as coordenadas das mãos na imagem
    :param img: imagem capturada pela câmera
    :return: imagem com as marcações das mãos e as coordenadas convertidas em píxeis.
    """

    # O módulo de detecção de mãos do MediaPipe espera uma imagem no formato RGB
    imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processa a imagem e retorna o resultado:
    # MULTI_HAND_LANDMARKS: informa as coordenadas normalizadas dos pontos da mão
    # MULTI_AND_WORLD_LANDMARKS: informa as coordenadas em metros
    # MULTI_HANDEDNESS: informa o lado da mão (esquerda/direita) + probabilidade de acerto
    resultado = maos.process(imagem_rgb)

    # Desenha as marcações das mãos na imagem, caso existam
    todas_maos = []
    if resultado.multi_hand_landmarks:

        for lado_mao, marcacao_maos in zip(
            resultado.multi_handedness, resultado.multi_hand_landmarks
        ):
            info_mao = {}
            coordenadas = []
            for marcacao in marcacao_maos.landmark:
                # Convertendo e armazenando os valores das coordenadas x, y e z
                coord_x, coord_y, coord_z = (
                    int(marcacao.x * WIDTH),
                    int(marcacao.y * HEIGHT),
                    int(marcacao.z * WIDTH),
                )
                coordenadas.append((coord_x, coord_y, coord_z))

            info_mao["coordenadas"] = coordenadas
            todas_maos.append(info_mao)
            mp_desenho.draw_landmarks(img, marcacao_maos, mp_maos.HAND_CONNECTIONS)

    return img, todas_maos


def dedos_levantados(mao: dict[str, list[tuple[int, int, int]]]) -> list[bool]:
    dedos = []
    for ponta_dedo in [8, 12, 16, 20]:
        verifica_dedo = False

        if mao["coordenadas"][ponta_dedo][1] < mao["coordenadas"][ponta_dedo - 2][1]:
            verifica_dedo = True

        dedos.append(verifica_dedo)

    return dedos


# Laço de repetição para capturar a imagem da câmera e exibir na tela
while True:
    sucesso, imagem = camera.read()

    # Espelha a imagem horizontalmente
    imagem = cv2.flip(imagem, 1)

    if not sucesso:
        break

    (imagem, todas_maos) = encontra_coordenadas_maos(imagem)

    if len(todas_maos) == 1:
        info_dedos_mao1 = dedos_levantados(todas_maos[0])
        print(info_dedos_mao1)

    cv2.imshow("Imagem", imagem)

    # Verifica se a tecla ESC (código 27) foi pressionada
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera a câmera e fecha todas as janelas
camera.release()
cv2.destroyAllWindows()
