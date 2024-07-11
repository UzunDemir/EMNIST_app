
############################################################################################

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

st.sidebar.write("[Uzun Demir](https://uzundemir.github.io/)") #[Github](https://github.com/UzunDemir)     [Linkedin](https://www.linkedin.com/in/uzundemir/)     
st.sidebar.write("[Github](https://github.com/UzunDemir)")
st.sidebar.write("[Linkedin](https://www.linkedin.com/in/uzundemir/)")
st.sidebar.title("Описание")
st.sidebar.divider()
st.sidebar.write(
        """
                     Эта приложка выполнена в рамках практической работы по модулю Computer Vision курса Machine Learning Advanced от Skillbox.
                     
                     1. Вначале была обучена модель распознавания рукописных цифр на базе MNIST (Modified National Institute of Standards and Technology database).
                     Точность на тестовой выборке датасета должна быть не ниже 68%. Я использовал много разных моделей и остановил свой выбор на сверточной нейронной сети (Convolutional Neural Network, CNN)
                     которая показала точность на тестовом наборе данных: 0.99.
                     Ноутбук с исследованиями можно посмотреть [здесь.](https://github.com/UzunDemir/mnist_777/blob/main/RESEARCH%26MODEL/prepare_model.ipynb)
                     2. Вторым шагом необходимо было обернуть готовую модель в сервис и запустить её как часть веб-приложения для распознавания самостоятельно написанных символов. 
                     После этого нужно было создать docker-образ и запустить приложение в docker-контейнере.
                     3. Я решил сделать [полноценное приложение, которое загружает изображение цифры и предсказывает ее](https://mnistpred.streamlit.app/). 
                     Но как злостный перфекционист, я подумал: а что если самому рисовать цифру и пусть модель ее предсказывает! 
                     Немного поискал как реализовать эту идею  и остановил свой выбор на Streamlit.
                     И вот что получилось!
                     
                     """
    )

# Устанавливаем стиль для центрирования элементов
st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        /height: 5vh;
        text-align: center;
        flex-direction: column;
        margin-top: 0vh;  /* отступ сверху */
    }
    .github-icon:hover {
        color: #4078c0; /* Изменение цвета при наведении */
    }
    </style>
    <div class="center">
        <img src="https://github.com/UzunDemir/mnist_777/blob/main/200w.gif?raw=true">
        <h1>Классификатор рукописных цифр</h1>
        <p>Напишите (пока) только одну цифру!</p>
    </div>
    """, unsafe_allow_html=True)
st.divider()
# Настройки для канвы
stroke_width = 10
stroke_color = "black"
bg_color = "white"
drawing_mode = "freedraw"

# Создаем две колонки
col1, col2 = st.columns([1, 1])

# В первой колонке создаем компонент квадратной канвы
with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки с некоторой прозрачностью
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=150,  # Установим высоту и ширину канвы одинаковыми (в пикселях)
        width=150,
        drawing_mode=drawing_mode,
        key="canvas",
    )



# Функция для загрузки модели с дополнительными проверками
def load_model_safe(model_path):
    try:
        with tf.keras.backend.name_scope('model_loading'):
            model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except IndexError as e:
        print(f"IndexError: {e}. Возможно, стек областей видимости пуст.")
        # Дополнительная обработка или попытка повторной загрузки модели
        return None
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

# Попытка загрузки модели
model = load_model_safe('model.h5')

if model is not None:
    print("Model loaded successfully.")
else:
    print("Failed to load the model.")


##############################
def preprocess_canvas_image(image_data):
    img = Image.fromarray(image_data)
    img = img.convert('L')  # Преобразование в оттенки серого
    resized_image = img.resize((28, 28), Image.BILINEAR)  # Изменение размера
    grayscaled_image = resized_image.convert("L")  # Преобразование в grayscale
    inverted_image = 255 - np.array(grayscaled_image)  # Инверсия цветов
    normalized_image = inverted_image / 255.0  # Нормализация значений
    reshaped_image = normalized_image.reshape((1, 28, 28, 1))  # Изменение формы для модели CNN

    st.write("Визуализация предобработки цифры")

    # Отображение предобработанных изображений с помощью Matplotlib
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

    # Скрытие меток осей для всех подграфиков
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    # Исходное изображение
    ax[0, 0].set_title("Original Image")
    ax[0, 0].imshow(img, cmap='gray')

    # Измененное изображение (28 * 28)
    ax[0, 1].set_title("Resized Image")
    ax[0, 1].imshow(resized_image, cmap='gray')

    # Grayscale изображение
    ax[0, 2].set_title("Grayscale Image")
    ax[0, 2].imshow(grayscaled_image, cmap="gray")

    # Инвертированное изображение (текст белый, фон черный)
    ax[1, 0].set_title("Inverted Image")
    ax[1, 0].imshow(inverted_image, cmap="gray")

    # Нормализованное изображение (делим на 255, чтобы значения были от 0 до 1)
    ax[1, 1].set_title("Normalized Image")
    ax[1, 1].imshow(normalized_image, cmap="gray")

    # Измененная форма изображения
    ax[1, 2].set_title("Reshaped Image")
    ax[1, 2].imshow(reshaped_image.reshape((28, 28)), cmap="gray")

    st.pyplot(fig)

    return reshaped_image



if st.button('Классифицировать нарисованную цифру'):
    try:
        # Предобработка изображения с холста
        img_array = preprocess_canvas_image(canvas_result.image_data)

        # Предсказание с использованием предварительно обученной модели
        result = model.predict(img_array)
        predicted_class = np.argmax(result)
    
        #st.success(f'Предсказанная цифра: {predicted_class}')
                # Во второй колонке выводим предсказанную цифру
        with col2:
            # Пример предсказанной цифры (замените на ваш результат)
            #predicted_class = 7
            st.success(f'Предсказанная цифра: {predicted_class}') 
    except Exception as e:
        st.error(f'Ошибка: {e}')



