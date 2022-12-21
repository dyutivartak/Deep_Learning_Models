import keras
import cv2
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential , load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train= x_train.reshape(60000,28,28)
x_test= x_test.reshape(x_test.shape[0],28,28)
input_shape=(28,28,1)
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)
x_train= x_train.astype('float32')
x_test= x_test.astype('float32')
for i in range(10):
    plt.imshow(x_train[i])
    plt.subplot(5,2,i+1)

x_train /= 255
x_test /=255
batch_size=64 
num_classes=10
epochs=2

def build_model(optimizer): 
    model=Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape)) 
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Dropout(0.25)) 
    model.add(Flatten()) 
    model.add(Dense(256, activation='relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(num_classes, activation='softmax')) 
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer= optimizer, metrics=['accuracy']) 
    model.summary()
    return model 

model = build_model('Adam') 
plot_model(model, to_file="mnist model"+ '.jpg', show_shapes=True)
hist=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test))

y1=hist.history['accuracy']
y2=hist.history['val_accuracy']
plt.plot(y1)
plt.plot(y2)

keras.models.save_model(model, "mnist.h5", save_format="h5")
def predict_image(model, img):
    img = np.reshape(img,(1, 28, 28))
    pred = model.predict(img)
    img = img.astype('float32')
    img /=255
    print(pred)
    answer = np.argmax(pred)
    print(answer)

m = load_model('/content/mnist.h5')
predict_image(m, x_test[7])
plt.imshow(x_test[7],cmap='gray')
