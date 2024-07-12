# Установка необходимых библиотек
!pip install streamlit opencv-python-headless

# Импорт библиотек
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from shiftlab_ocr.doc2text.reader import Reader

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
    img = Image.fromarray(image_data)
    img = img.convert('L')  # Преобразование в оттенки серого
    img = img.resize((28, 28), Image.BILINEAR)  # Изменение размера
    img_array = np.array(img)
    return img_array

# Проверка, если кнопка нажата
if st.button('Распознать текст'):
    # Предобработка изображения с канвы
    if canvas_result.image_data is not None:
        img_array = preprocess_image(canvas_result.image_data)

        # Сохранение изображения
        cv2.imwrite('handwritten_text.png', img_array)

        # Использование модели для распознавания текста
        reader = Reader()
        result = reader.doc2text('handwritten_text.png')

        # Вывод распознанного текста
        st.write("Распознанный текст:")
        st.write(result[0])
    else:
        st.write("Пожалуйста, нарисуйте текст перед распознаванием.")
