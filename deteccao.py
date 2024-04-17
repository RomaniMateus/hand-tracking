import os

import cv2
import mediapipe as mp
import numpy as np

# Definição de constantes
WIDTH = 1920  # Largura da imagem
HEIGHT = 1080  # Altura da imagem
OFFSET = 50  # Offset para centralizar o teclado
BRANCO = (255, 255, 255)  # Cor branca
AZUL = (255, 0, 0)  # Cor azul
VERDE = (0, 255, 0)  # Cor verde
VERMELHO = (0, 0, 255)  # Cor vermelho
PRETO = (0, 0, 0)  # Cor preto
TECLAS = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", " "],
]

# Definindo variáveis de checagem para abrir e fechar os aplicativos
bloco_notas_aberto = False
chrome_aberto = False
calculadora_aberta = False

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
    todas_maos_info = []
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
            info_mao["lado"] = lado_mao.classification[0].label

            todas_maos_info.append(info_mao)
            mp_desenho.draw_landmarks(img, marcacao_maos, mp_maos.HAND_CONNECTIONS)

    return img, todas_maos_info


def dedos_levantados(mao: dict[str, list[tuple[int, int, int]]]) -> list[bool]:
    """
    Função que verifica quais dedos estão levantados (True) ou abaixados (False). Nesta função não consideramos o
    polegar.
    :param mao: dicionário com as coordenadas dos pontos de referência da mão
    :return: lista com os dedos
    levantados (True) ou abaixados (False)
    """
    dedos = []

    # [8, 12, 16, 20] -> pontas dos dedos indicador, médio, anelar e mínimo
    for ponta_dedo in [8, 12, 16, 20]:
        verifica_dedo = False

        # Verifica se a coordenada y da ponta do dedo é MENOR que a coordenada y da base do dedo.
        # OBS.: No openCV a origem do eixo y é o canto superior esquerdo da imagem, por isso a comparação é com MENOR.
        if mao["coordenadas"][ponta_dedo][1] < mao["coordenadas"][ponta_dedo - 2][1]:
            verifica_dedo = True

        dedos.append(verifica_dedo)

    return dedos


def imprime_botoes(
    img: np.ndarray,
    posicao: tuple[int, int],
    letra: str,
    tamanho: int = 50,
    cor_retangulo: tuple[int, int, int] = BRANCO,
):

    cv2.rectangle(
        img,
        posicao,
        (posicao[0] + tamanho, posicao[1] + tamanho),
        cor_retangulo,
        cv2.FILLED,
    )
    cv2.rectangle(img, posicao, (posicao[0] + tamanho, posicao[1] + tamanho), AZUL, 1)
    cv2.putText(
        img,
        letra,
        (posicao[0] + 15, posicao[1] + 30),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        PRETO,
        2,
    )

    return img


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

        if todas_maos[0]["lado"] == "Left":
            for indice_linha, linha_teclado in enumerate(TECLAS):
                for indice, letra in enumerate(linha_teclado):
                    img = imprime_botoes(
                        imagem,
                        (OFFSET + indice * 80, OFFSET + indice_linha * 80),
                        letra,
                    )
        if todas_maos[0]["lado"] == "Right":
            # Abrindo o Bloco de Notas, Chrome e Calculadora conforme a posição dos dedos
            if (
                info_dedos_mao1 == [True, False, False, False]
                and not bloco_notas_aberto
            ):
                bloco_notas_aberto = True
                os.startfile(r"C:\WINDOWS\system32\notepad.exe")
            if info_dedos_mao1 == [True, True, False, False] and not chrome_aberto:
                chrome_aberto = True

                try:
                    os.startfile(
                        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
                    )
                except FileNotFoundError:
                    os.startfile(
                        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
                    )

            if info_dedos_mao1 == [True, True, True, False] and not calculadora_aberta:
                calculadora_aberta = True
                os.startfile(r"C:\WINDOWS\system32\calc.exe")

            # Fechando o Bloco de Notas quando todos os dedos estiverem abaixados
            if info_dedos_mao1 == [False, False, False, False] and bloco_notas_aberto:
                bloco_notas_aberto = False
                os.system("TASKKILL /F /IM notepad.exe")

            # Fechando a janela de imagem do openCV com os dedos indicador e mínimo levantados
            if info_dedos_mao1 == [True, False, False, True]:
                break
    cv2.imshow("Imagem", imagem)

    # Verifica se a tecla ESC (código 27) foi pressionada
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera a câmera e fecha todas as janelas
camera.release()
cv2.destroyAllWindows()
