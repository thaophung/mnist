from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import argparse
import sys
import cv2

def main(image):
    model = Sequential()
    #model.add(Flatten(input_shape=(28,28)))
    input_shape=(28,28,1)
    num_classes=10
    model.add(Conv2D(32, kernel_size=(3,3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights('mnist_model.h5')

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta())
    im_pred = list(model.predict(image)[0])
    #print(im_pred) 

    output = im_pred.index(1.0)
    print(output)

if __name__ == "__main__":
    image = sys.argv[1]
    #print(image)
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.reshape(1,28,28,1)
    #print(image.shape)
    main(image)
        
