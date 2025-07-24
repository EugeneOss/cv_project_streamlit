import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ultralytics import YOLO
from io import StringIO
import cv2
import requests
from PIL import Image
from io import BytesIO

# Приложение будет принимать картинку человека/людей и брюлить им лица.

# 0. Модуль кода для работы самого скрипта и модели

model_final = YOLO("./models/model_2/best_2.pt")

# 1. Делаем описание проекта.
st.badge('🚨 powered by EugeneOss 🚨')
st.title('Данная модель позволяет размыть лица на любом фото, которые вы добавляете через URL или посредством загрузки файла')

st.link_button("Ваш модельер", "https://t.me/Eugene_Os")



# 2. добавляем кнопку загрузки файла

option_map = {
    0: "Загрузи фото",
    1: "Сфотографируй сразу",
    2: "Загрузи по ссылку",
}

st.write('''
## Выбери метод доставки изображения в модель
''')

selection = st.segmented_control(
    "",
    options=option_map.keys(),
    format_func=lambda option: option_map[option],
    selection_mode="single",
)

if selection == 0:
    browse = st.file_uploader('Выбери картинку', type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if browse is not None:
        with open("saved_image.png", "wb") as f:
            f.write(browse.read())
        image = cv2.imread("saved_image.png")
        results = model_final(image)[0]
        blurred_img = image.copy()
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            face_roi = blurred_img[y1:y2, x1:x2]
            face_blur = cv2.GaussianBlur(face_roi, (35, 35), 30)
            blurred_img[y1:y2, x1:x2] = face_blur
        st.image(blurred_img[:, :, ::-1])
        
        _, buffer = cv2.imencode(".jpg", blurred_img)
        img_bytes = BytesIO(buffer.tobytes())

        st.download_button(
            label='Скачать изображение',
            file_name='faces_blurred.jpg',
            mime="image/jpeg",
            data=img_bytes
            )

elif selection == 1:
    checkbox = st.checkbox('Доступ к камере')
    if checkbox is True:
        browse = st.camera_input('Сфотографировать')
        if browse is not None:
            with open("saved_image.png", "wb") as f:
                f.write(browse.read())
            image = cv2.imread("saved_image.png")
            results = model_final(image)[0]
            blurred_img = image.copy()
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                face_roi = blurred_img[y1:y2, x1:x2]
                face_blur = cv2.GaussianBlur(face_roi, (35, 35), 30)
                blurred_img[y1:y2, x1:x2] = face_blur
            st.image(blurred_img[:, :, ::-1])

            _, buffer = cv2.imencode(".jpg", blurred_img)
            img_bytes = BytesIO(buffer.tobytes())

            st.download_button(
            label='Скачать изображение',
            file_name='faces_blurred.jpg',
            mime="image/jpeg",
            data=img_bytes
            )

elif selection == 2:
    urls = st.text_input('Введите ссылку')
    if urls:
        try:
            response = requests.get(urls, timeout=5)
            # Проверка статуса и Content-Type
            content_type = response.headers.get("Content-Type", "")
            if "image" in content_type:
                browse = Image.open(BytesIO(response.content))
                with open("saved_image.png", "wb") as f:
                    f.write(response.content)
                image = cv2.imread("saved_image.png")
                results = model_final(image)[0]
                blurred_img = image.copy()
                for box in results.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)
                    face_roi = blurred_img[y1:y2, x1:x2]
                    face_blur = cv2.GaussianBlur(face_roi, (35, 35), 30)
                    blurred_img[y1:y2, x1:x2] = face_blur
                st.image(blurred_img[:, :, ::-1])

                _, buffer = cv2.imencode(".jpg", blurred_img)
                img_bytes = BytesIO(buffer.tobytes())

                st.download_button(
                label='Скачать изображение',
                file_name='faces_blurred.jpg',
                mime="image/jpeg",
                data=img_bytes
                )
            else:
                st.error("🚫 Это не изображение. Убедитесь, что ссылка ведёт на файл с расширением .jpg, .png и т.д.")
        except Exception as e:
            st.error(f"❌ Не удалось загрузить изображение: {e}")


