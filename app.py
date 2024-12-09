import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#Cargar el modelo previamente entrenado
model = tf.keras.models.load_model(r"C:\Users\LAPTOP\OneDrive\Escritorio\UdeA\Ingenieria de materiales\Celdas de combustible\Python\Talento-Tech\mlp_modelo_h5")

#Función para preprocesar la imagen
def preprocess_image(image):
    # Convertir la imagen a escala de grises y redimensionar a 12x12
    image= image.resize((12, 12))#redimensionar 
    image = np.array(image) #conertir a array de numpy
    image = image / 255.0 #normalizar a valores entre 0 y 1
    image = np.reshape(image,(1,28*28))# Redimensionar para la entrada del modelo
    return image
 
# Título de la aplicación
st.title('Espesor pelicula Celda solar Perovskita')
# Cargar la imagen
uploaded_file = st.file_uploader("Cargar una imagen", type=['png', 'jpg','jpeg'])
if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen Cargada",use_column_width=True)
    # preprocesar la imagen
    processed_image= preprocess_image(image)
    # Hacer la predicción
    prediction = model.predict(processed_image)
    predicted_digit =np.argmax(prediction)
    # Mostrar la predicción
    st.write(f"Predicción: **{predicted_digit}**")
    # Mostrar probabilidades
    # Mostrar imagen procesada para ver el preprocesamiento
    #plt.imshow(np.squeeze(processed_image),cmap='gray')
    #plt.axis('off')
    #st.pyplot(plt)
    
    plt.imshow(processed_image.reshape(12, 12)) # Convierte a 28x28 para visualizar
    plt.axis('off')
    st.pyplot(plt)