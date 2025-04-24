import cv2
import numpy as np
from tensorflow.keras.models import load_model 

# Carrega o modelo de detecção de máscara
model = load_model("mask_detector.h5")  # Substitua com o caminho correto

# Carrega o classificador de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Carrega a imagem PNG com transparência (4 canais)
image = cv2.imread("foto1.png", cv2.IMREAD_UNCHANGED)

# Se for uma imagem com 3 canais apenas, adiciona canal alfa
if image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

original = image.copy()

# Converte para cinza para detectar rostos
gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

# Detecta rostos na imagem
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

# Cria uma imagem de saída transparente
output = np.zeros_like(image)

for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w, :3]
    face_resized = cv2.resize(face, (224, 224)) / 255.0
    face_array = np.expand_dims(face_resized, axis=0)

    # Faz a predição: [máscara, sem máscara]
    (mask, no_mask) = model.predict(face_array)[0]

    if mask > no_mask:
        # Copia a região da imagem original com alfa = 255
        person = original[y:y+h, x:x+w]
        output[y:y+h, x:x+w] = person

# Salva a imagem final em PNG com fundo transparente
cv2.imwrite("pessoas_com_mascara.png", output)
