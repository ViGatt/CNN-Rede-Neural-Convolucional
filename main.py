import tensorflow as tf
import numpy as np
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt


# Caminho para a pasta de dados local
train_dir = "data/train" 
validation_dir = "data/test"
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

# Criar gerador de dados para treinamento e validação

train_datagenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagenerator = ImageDataGenerator(rescale=1./255)

datagenerator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

treinamento_dataset = train_datagenerator.flow_from_directory(
    directory=train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validacao_dataset = validation_datagenerator.flow_from_directory(
    directory=validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

print("Classes encontradas:", treinamento_dataset.class_indices)

modelo = Sequential([
    # Camada de Convolução 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    # Camada de Convolução 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Camada de Achatamento e Camadas Densas
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Camada de Dropout para combater overfitting
    Dense(treinamento_dataset.num_classes, activation='softmax') 
])
# Compilar o modelo
modelo.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = modelo.fit(
    treinamento_dataset,
    steps_per_epoch=len(treinamento_dataset),
    epochs=epochs,  # Usando a variável que você já definiu
    validation_data=validacao_dataset,
    validation_steps=len(validacao_dataset)
)
caminho_nova_imagem = 'data/test/Soybean_healthy/001.jpg'
nova_imagem = tf.keras.preprocessing.image.load_img(caminho_nova_imagem, target_size=(img_height, img_width))
nova_imagem = tf.keras.preprocessing.image.img_to_array(nova_imagem) / 255.0
nova_imagem = np.expand_dims(nova_imagem, axis=0)

previsao = modelo.predict(nova_imagem)
classe_predita = np.argmax(previsao)
classes = list(treinamento_dataset.class_indices.keys())

print(f"Classe prevista: {classes[classe_predita]} com probabilidade {previsao[0][classe_predita]:.2f}")

# Visualizar curvas de acurácia e perda
plt.figure(figsize=(12, 5))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda')
plt.legend()

plt.show()