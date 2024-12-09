import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf

# Rutas
ruta_labels = r"C:\Users\LAPTOP\OneDrive\Escritorio\UdeA\Ingenieria de materiales\Celdas de combustible\Python\Talento-Tech\DatasetRichi\dataset.csv"
ruta_imagenes = r"C:\Users\LAPTOP\OneDrive\Escritorio\UdeA\Ingenieria de materiales\Celdas de combustible\Python\Talento-Tech\DatasetRichi\imagenes\\"

# Leer etiquetas desde CSV
labels_df = pd.read_csv(ruta_labels)
image_paths = [ruta_imagenes + img_name for img_name in labels_df["Archivo"]]
labels = labels_df["Espesores Promedio"]

# Función para cargar y preprocesar las imágenes
def load_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [12, 12])
    img = img / 255.0
    img = tf.reshape(img, [-1])  # Aplana la imagen
    return img, tf.cast(label, tf.float32)

# Crear dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_image).shuffle(len(image_paths)).batch(1)

# Visualización
def visualizar_normalizado(dataset, num_muestras=1):
    plt.figure(figsize=(10, 10))
    for batch_imagenes, batch_etiquetas in dataset.take(1):  # Toma un lote
        for i in range(min(num_muestras, tf.shape(batch_imagenes)[0])):
            imagen = batch_imagenes[i]
            etiqueta = batch_etiquetas[i]
            imagen = tf.clip_by_value(imagen * 255, 0, 255)  # Desnormalizar si es necesario
            plt.subplot(1, num_muestras, i + 1)
            plt.imshow(tf.reshape(imagen, [12, 12, 3]).numpy().astype("uint8"))
            plt.title(f"{etiqueta.numpy(): .3f}")
            plt.axis("off")
        plt.show()
        break

visualizar_normalizado(dataset)

# Dividir el dataset
total_size = len(image_paths)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

# Batching y prefetching
train_dataset = train_dataset.shuffle(train_size).batch(1).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(1).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(1).prefetch(tf.data.AUTOTUNE)

# Confirmar tamaños
print(f"Size del conjunto de entrenamiento: {train_size}")
print(f"Size del conjunto de validación: {val_size}")
print(f"Size del conjunto de prueba: {test_size}")

# Definir modelo
model = models.Sequential([
    layers.Input(shape=(12 * 12 * 3,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')  # Salida para regresión
])

# Compilar modelo
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

# Entrenar modelo
history = model.fit(train_dataset, epochs=5, validation_data=val_dataset)

# Evaluar modelo
test_loss, test_mae = model.evaluate(test_dataset)
print(f'Error absoluto medio en el conjunto de prueba: {test_mae}')

model.save('mlp_modelo_espesor.keras')
