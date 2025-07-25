import streamlit as st
import torch
import torch.nn as nn
from torch.nn.functional import relu
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import io
import numpy as np
import requests
import tempfile
import os

# 1. Создаем класс для модели

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
    
# 2. назначаем device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Инициализация
model1 = UNet(n_class=2).to(device)

# 4. Загрузка весов
model1.load_state_dict(torch.load('./models/best.pth', map_location=device))

@st.cache_resource
def load_model_from_hf():
    model_url = "https://huggingface.co/EugeneOss/models_for_cv_project/resolve/main/best.pth"
    local_path = "./models/best.pth"

    # Если модель ещё не скачана — скачиваем
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            response = requests.get(model_url)
            f.write(response.content)

    model = UNet(n_class=2).to(device)
    model.load_state_dict(torch.load(local_path, map_location=device))
    model.eval()  # переводим в режим предсказания

    return model


# Используем
model1 = load_model_from_hf()

st.title("Данное приложение позволяет маскировать леса на аэрофотоснимках")



# Трансформация (resize до 256x256 и в тензор)
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

image = st.file_uploader('Загрузи Аэрофотоснимок', type=["jpg", "jpeg", "png"], accept_multiple_files=False)
if image is not None:

    # Загрузка и преобразование
    img = Image.open(image).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 256, 256]

    # Предсказание
    with torch.no_grad():
        output = model1(input_tensor)  # [1, C, 256, 256]
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu()  # [256, 256]

    # resize для PIL
    original = img.resize((256, 256))
    predicted = Image.fromarray(np.uint8(pred_mask.numpy() * 255))  # если классы 0/1

    st.image([original, predicted], caption=["Оригинал", "Маска"], width=256)


