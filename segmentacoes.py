import cv2
import numpy as np
import os

# Caminho da imagem original
image_path = "foto1.png"
original = cv2.imread(image_path)
if original is None:
    raise FileNotFoundError("Imagem 'foto1.png' não encontrada.")

# Criação da pasta de saída
output_dir = "imagens segmentacao"
os.makedirs(output_dir, exist_ok=True)

# 1️⃣ Thresholding (aplicado em escala de cinza)
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(output_dir, "foto1_thresholding.png"), thresh)

# 2️⃣ Segmentação por cor (HSV) – Ex: segmentar cor azul
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 150, 0])  # valores HSV de azul
upper_blue = np.array([140, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
segmented = cv2.bitwise_and(original, original, mask=mask_blue)
cv2.imwrite(os.path.join(output_dir, "foto1_segmentacao_cor.png"), segmented)

# 3️⃣ Detecção de bordas (Canny)
edges = cv2.Canny(original, threshold1=100, threshold2=200)
cv2.imwrite(os.path.join(output_dir, "foto1_bordas_canny.png"), edges)

# 4️⃣ Anotação com bounding boxes (sobre faces como exemplo)
# Usando Haar Cascade (pré-treinado) para detectar rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

annotated = original.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(annotated, "Rosto", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite(os.path.join(output_dir, "foto1_anotada.png"), annotated)

print("✅ Técnicas de segmentação e detecção aplicadas com sucesso!")
