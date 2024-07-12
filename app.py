# Установка необходимых библиотек
# !pip install transformers
# !pip install torch
# !pip install streamlit opencv-python-headless

# Импорт библиотек
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Загрузка модели и процессора TrOCR
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Создание интерфейса Streamlit
st.title("Распознавание рукописного текста")
st.write("Пожалуйста, напишите текст в поле ниже:")

# Создание канвы для рисования
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки с некоторой прозрачностью
    stroke_width=2,
    stroke_color="black",
    background_color="white",
    height=150,  # Установим высоту и ширину канвы одинаковыми (в пикселях)
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Функция для предобработки изображения
def preprocess_image(image_data):
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('L')
    img = img.resize((384, 384), Image.BILINEAR)  # Изменение размера
    img_array = np.array(img)
    return img_array

# Проверка, если кнопка нажата
if st.button('Распознать текст'):
    # Предобработка изображения с канвы
    if canvas_result.image_data is not None:
        img_array = preprocess_image(canvas_result.image_data)

        # Сохранение изображения
        img = Image.fromarray(img_array)
        img.save('handwritten_text.png')

        # Использование модели для распознавания текста
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Вывод распознанного текста
        st.write("Распознанный текст:")
        st.write(generated_text)
    else:
        st.write("Пожалуйста, нарисуйте текст перед распознаванием.")
