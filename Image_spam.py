from keras.models import Sequential
from keras.layers.core import Dense , Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.np_utils import to_categorical
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
#scipy
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

#data
path1 = '/home/abhishek/Desktop/image_data'
path2 = '/home/abhishek/Desktop/input_data'

listing = os.listdir(path1)

num_sample = size(listing)
print num_sample
img_rows, img_cols = 200, 200
valid_imags = [".jpg",".gif", ".png", ".jpeg", ".com_", ".bmp",".edu_", ".tif", ".edu_serv"]
for file in listing:
    ext = os.path.splitext(file)[1]
    if ext.lower() not in valid_imags:
        print (ext)
        continue
    im = Image.open(path1 + '/' +file)
    img = im.resize((img_rows, img_cols))
    gray = img.convert('L')
    
    gray.save(path2 + '/' + file, 'JPEG')
    
imlist = os.listdir(path2)
imlist = np.sort(imlist)
im1 = array(Image.open(path2 + '/' + imlist[1]))
m,n = im1.shape[0:2]
imnbr = len(imlist)

#create matrix to store all flatten images
immatrix = array([array(Image.open(path2 + '/' + im2)).flatten()
                for im2 in imlist], 'f')
label = np.ones((num_sample,),dtype = int)
label[0:398] = 1
label[398:812] = 0

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

img = immatrix[0].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img, cmap='gray')

print(train_data[0].shape)
print(train_data[1].shape)

X_train, X_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size = 0.3, random_state = 4)

X_train = X_train.reshape(-1, img_rows*img_cols)
X_test = X_test.reshape(-1,img_rows*img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /=255

#convert class vector to binary matrices
Y_train = to_categorical(y_train, nb_classes=2)
Y_test = to_categorical(y_test, nb_classes=2)

i=100
plt.imshow(X_train[i], interpolation='nearest')
print('label:', Y_train[i,:])



#model
model = Sequential()



model.add(Dense(128, input_dim = 200*200, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

h = model.fit(X_train, Y_train, nb_epoch = 30,validation_split = 0.3)

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')

model.evaluate(X_test, Y_test)

y_pred = model.predict(X_test)
y_pred[:5]

y_test_class = np.argmax(Y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_test_class, y_pred_class))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_class, y_pred_class)
