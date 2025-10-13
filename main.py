import tensorflow as tf
import numpy as np
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Caminho para a pasta de dados local
dataset_dir = "dados" 
img_height, img_width = 224, 224
batch_size = 32

# Remover pastas desnecessárias que podem atrapalhar o gerador de imagens
for root, dirs, files in os.walk(dataset_dir):
    for dir_name in dirs:
        if dir_name == ".ipynb_checkpoints":
            shutil.rmtree(os.path.join(root, dir_name))
            print(f"Removido: {os.path.join(root, dir_name)}")

# Criar gerador de dados para treinamento e validação
datagenerator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

treinamento_dataset = datagenerator.flow_from_directory(
    directory=dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training' 
)

validacao_dataset = datagenerator.flow_from_directory(
    directory=dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation' 
)

print("Classes encontradas:", treinamento_dataset.class_indices)