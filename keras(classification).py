import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.utils import to_categorical
from keras.datasets import mnist

import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print(X_train.shape)
plt.imshow(X_train[1], cmap='gray')
num_pixels = X_train.shape[1] * X_train.shape[2] 
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images
#print(X_train.shape[1])
X_train=X_train / 255
X_test=X_test / 255
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
#you can take y_test or y_train
num_classes=y_train.shape[1]
#print(num_classes)


##################################################3
def classification_model():
  model=Sequential()
  model.add(Input(shape=(num_pixels,)))
  model.add(Dense(num_pixels,activation="relu"))
  model.add(Dense(100,activation="relu"))
  model.add(Dense(num_classes,activation="softmax"))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model
model = classification_model()

early_stop=EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=2, callbacks=[early_stop])
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: {:.2f}%'.format(scores[1] * 100))       
model.save('classification_model.keras')
pretrained_model = keras.saving.load_model('classification_model.keras')
