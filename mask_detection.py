from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

#Limpiar algunos warnings
import warnings
warnings.filterwarnings("ignore")


def mask_detection(frame, faceNet, maskNet):
	# De acuerdo a la dimension del video se sca la porcion (blob)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# Pasamos la porcion a traves de la red para que sea evaluada
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

	# inicializamos una lista que guardara los rostros, sus localizaciones y prediccion
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		# Extraemos la confiabilidad de la deteccion
		confidence = detections[0, 0, i, 2]

		# Filtramos las predicciones no tan confiables
		if confidence > 0.5:
			# Obtenemos las dimensiones del area a reconocer
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Nos aseguramos que las dimensiones no salgan del frame del video
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Convertimos la imagen a rgb
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Añadimos los rostros a su respectiva lista
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Solo hacemos predicciones o reconocimiento si hay un rostro
	# Esto con el fin de optimizar el programa
	if len(faces) > 0:
		# Para acelerar se hara reconocimiento de todos los rostros al mismo tiempo
		# en vez de hacerlo por separado
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

# Cargamos nuestros archivos para la deteccion
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargamos el model generado a partir del entrenamiento
trained_model = load_model("mask_detector.model")

# Inicializacion del video
vs = VideoStream(src=0).start()

# While infinito donde se procesa toda la informacion de la camara
while True:
	# Redimensionamos el tamaño de la camara
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	# Usa la funcion para detectar mascarillas
	(locs, preds) = mask_detection(frame, faceNet, trained_model)

	# For para detectar la zona del video en donde se produce el reconocimiento
	for (box, pred) in zip(locs, preds):
		# Sacamos las posiciones de las 4 esquinas
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Cambiamos el color y etiqueta de acuerdo a lo reonocido
		# Rojo - sin mascarilla
		# Verde - con mascarilla
		label = "Mask" if mask < withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Mostramos la precision con 2 decimales
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Funciones para imprimir todo lo anterior en el video

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Impresion de cada frame del video en tiempo real
	cv2.imshow("Mask Detector", frame)
	key = cv2.waitKey(1) & 0xFF

	# Condicional
	# Si se presiona la letra "q" se termina el loop del video
	if key == ord("q"):
		break

# Cerramos las ventanas y video
cv2.destroyAllWindows()
vs.stop()