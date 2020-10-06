# USAGE
# python mnist_runner.py


from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD

from models import Simplenn

######################################################################################
#                 Dataset preprocessing                                              #
######################################################################################

print("[INFO] accessing MNIST...")
((train_images, train_labels), (testX, testY)) = mnist.load_data()

# scale data to the range of [0, 1]
train_images = train_images.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# sample out validation data from trainX
(trainX, valX, trainY, valY) = train_test_split(train_images, train_labels,
                                                test_size=0.2, stratify=train_labels, random_state=42)
print("===========================================================================================")
print("[INFO] : Shapes of the raw data ")
print("=====> Train Data            ", trainX.shape, trainY.shape)
print("=====> Validation Data       ", valX.shape, valY.shape)
print("=====> Test Data             ", testX.shape, testY.shape)

# if we are using a simple neural network then we need to
# flatten the data to 28 * 28 * 1
# for the convolutional neural network RGB normalization is enough
trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
valX = valX.reshape((valX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28 * 28 * 1))

print("===========================================================================================")
print("[INFO] : Shapes of the data after reshaping")
print("=====> Train Data            ", trainX.shape, trainY.shape)
print("=====> Validation Data       ", valX.shape, valY.shape)
print("=====> Test Data             ", testX.shape, testY.shape)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
valY = lb.transform(valY)
testY = lb.transform(testY)

print("===========================================================================================")
print("[INFO] : Shapes of the data after binarizer")
print("=====> Train Data            ", trainX.shape, trainY.shape)
print("=====> Validation Data       ", valX.shape, valY.shape)
print("=====> Test Data             ", testX.shape, testY.shape)

######################################################################################
#                 Hyperparameter initialization                                      #
######################################################################################
input_shape = trainX.shape
n_classes = 10
n_layers = [64, 32, 32, 10]
activation_type = {"input": "relu", "hidden": "relu", "output": "softmax"}
learning_rate = 0.01
n_epochs = 100

now = datetime.now()
timestamp = now.strftime("%d%m%y%H%M%S")
output_folder = "outputs/mnist"
plt.rcParams['font.size'] = 9

######################################################################################
#                 Model initialization                                               #
######################################################################################
print("[INFO] building and training network...")

opt = SGD(learning_rate)
model = Simplenn.build(input_shape, n_classes, n_layers, activation_type)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(valX, valY),
              epochs=n_epochs, batch_size=128)
model.summary()

######################################################################################
#                 Training Validation Loss/Accuracy plot                             #
######################################################################################

# plot the training loss and accuracy
plt.style.use("seaborn")
plt.figure(figsize=(14, 10))
plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

output_file = "mnist-val-train-" + timestamp + ".png"
path = output_folder + "/" + output_file
plt.savefig(path)

######################################################################################
#                 Model evaluation                                                   #
######################################################################################
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)

cr = classification_report(testY.argmax(axis=1),
                           predictions.argmax(axis=1),
                           target_names=[str(x) for x in lb.classes_])
print(cr)
plt.figure(figsize=(15, 10))

######################################################################################
#                 Confusion Matrix                                                   #
######################################################################################
cm = confusion_matrix(y_target=testY.argmax(axis=1),
                      y_predicted=predictions.argmax(axis=1),
                      binary=False)
plt.figure(figsize=(14, 10))
fig, ax = plt.subplots(1)
p = sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt='.2f', square=True, ax=ax)
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.title("Confusion Matrix")
output_file = "mnist-con-matrix-" + timestamp + ".png"
path = output_folder + "/" + output_file
plt.savefig(path)
