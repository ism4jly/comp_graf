import cv2
import numpy as np
import os

# Caminho da imagem original
image_path = "foto1.png"
original = cv2.imread(image_path)
if original is None:
    raise FileNotFoundError("Imagem 'foto1.png' não encontrada.")

# Cria a pasta 'imagens alteradas' se não existir
output_dir = "imagens alteradas"
os.makedirs(output_dir, exist_ok=True)

# 1- Redimensionamento da imagem colorida
resized = cv2.resize(original, (640, 480))
cv2.imwrite(os.path.join(output_dir, "foto1_redimensionada.png"), resized)

# 2- Conversão para escala de cinza
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_dir, "foto1_cinza.png"), gray)

# 3- Ajuste de brilho na imagem colorida
brightness = 30  # Pode ser negativo
bright = cv2.convertScaleAbs(original, alpha=1.0, beta=brightness)
cv2.imwrite(os.path.join(output_dir, "foto1_brilho.png"), bright)

# 4- Ajuste de contraste na imagem colorida
contrast = 30
alpha = contrast / 127 + 1
contrast_img = cv2.convertScaleAbs(original, alpha=alpha, beta=0)
cv2.imwrite(os.path.join(output_dir, "foto1_contraste.png"), contrast_img)

print("✅ Imagens processadas e salvas na pasta 'imagens alteradas'")
