from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Inicializamos las epocas y el batch size (lotes)
INIT_LR = 1e-4
EPOCHS = 20 #20
BS = 32 #32

DIRECTORY = r"C:\Users\MIGUEL\PycharmProjects\MaskPrediction\dataset"
CATEGORIES = ["with_mask", "incorrect_mask"]

# Juntamos las carpetas de imagenes y las guardamos en sus respectivas listas

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# Se binarizan las etiquetas (one hot enconding)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

#Escogemos los datos para entrenar
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)

# Se generan mas imaganes para mejorar el reocnocimiento
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Cargamos MobileNetV2 de tensorflow
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Declaramos el Model
model = Model(inputs=baseModel.input, outputs=headModel)

#Recorremos las capas del MOdel y las detenemos para que no se actualicen a cada rato
for layer in baseModel.layers:
    layer.trainable = False

# Compilamos el Model
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS) # lr por learning_rate
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# Entrenamiento
H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
              steps_per_epoch=len(trainX) // BS,
              validation_data=(testX, testY),
              validation_steps=len(testX) // BS,
              epochs=EPOCHS)

# Se crean las predicciones/reconocimiento del modelo
predIdxs = model.predict(testX, batch_size=BS)

# Para cada imagen se carga su respectiva etiqueta(label)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

# Guardamos el model para no tener que entrenar la red cada que se compile el codigo
model.save("mask_detector.model", save_format="h5")

# Grafico sobre la precision del modelo
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
