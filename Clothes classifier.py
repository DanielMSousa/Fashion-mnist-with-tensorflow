import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

#tensorflow version
print(tf.__version__)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalização:

X_train = X_train/255.0

X_test = X_test/255.0

# Redes neurais utilizam formato de vetor

#O -1 indica todos os elementos
X_train = X_train.reshape(-1, 28*28)

X_test = X_test.reshape(-1, 28*28)
# Cada um dos 784 pixels será mandado como entrada para a rede neural.


# # Construindo a rede neural:
model = tf.keras.models.Sequential()

print(model)

# Adicionando a primeira camada densa (fully-connected)
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))

#Adicionando camada de dropout para zerar 20% dos neurônios da camada.
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=(128, )))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

#Visualizando estrutura da rede neural
print(model.summary())

#Treinando o modelo
model.fit(X_train, y_train, epochs=10)

#Avaliando desempenho do modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Test accuracy: {}'.format(test_accuracy))

print('Test loss: {}'.format(test_loss))