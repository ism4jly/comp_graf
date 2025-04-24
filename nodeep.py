import cv2
import numpy as np

def detectar_rostos(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rostos = face_cascade.detectMultiScale(gray, 1.3, 5)
    return rostos

def tem_mascara(face_crop):
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    altura = hsv.shape[0]
    parte_inferior = hsv[altura//2:, :]  # Parte da boca para baixo
    saturacao_media = np.mean(parte_inferior[:, :, 1])

    print(f"Saturação média inferior: {saturacao_media:.2f}")  # 👈 debug

    # Critério simples: máscara = baixa saturação (azul/branca/cinza)
    return saturacao_media < 70

def remover_fundo_pessoas_com_mascara(imagem_path):
    img = cv2.imread(imagem_path)
    img_resultado = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

    rostos = detectar_rostos(img)
    print(f"{len(rostos)} rosto(s) detectado(s).")

    for i, (x, y, w, h) in enumerate(rostos):
        face = img[y:y+h, x:x+w]
        cv2.imwrite(f"debug_face_{i}.png", face) 

        if tem_mascara(face):
            print(f"[{i}] Máscara detectada em: x={x}, y={y}, w={w}, h={h}")
            pessoa = img[y:y+h, x:x+w]
            alpha = np.ones((h, w), dtype=np.uint8) * 255
            b, g, r = cv2.split(pessoa)
            rgba = cv2.merge((b, g, r, alpha))
            img_resultado[y:y+h, x:x+w] = rgba
        else:
            print(f"[{i}] Sem máscara detectada")

    cv2.imwrite("pessoas_com_mascara_sem_dl.png", img_resultado)
    print("Imagem gerada: pessoas_com_mascara_sem_dl.png")

# Executar
remover_fundo_pessoas_com_mascara("foto1.png")
