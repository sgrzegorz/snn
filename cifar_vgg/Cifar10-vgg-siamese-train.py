"""
- Siamese VGG, (library implementation) for Cifar 10
"""



import tensorflow as tf
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras import *
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
import random
import utils


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

Y_train = y_train
Y_test = y_test


num_classes = 10 
x_train = x_train/255.
y_train = utils.to_categorical(y_train, num_classes) 

x_test = x_test/255.
y_test = utils.to_categorical(y_test, num_classes) 

##########################################Functions to rename vgg19 model ###################################

def add_infix(a,b,character):  
    b.startswith(a)
    tail = b.split(a, 1)[1]
    b = a + character + tail
    return b

add_infix('block1_conv1','block1_conv1_ib-0','A')

def rename_network_nodes(vgg19,character):  # https://stackoverflow.com/questions/63373692/how-to-rename-the-layers-of-a-keras-model-without-corrupting-the-structure
    layer_names = {layer._name for layer in vgg19.layers}
    _network_nodes = []
    for b in vgg19._network_nodes:
        for a in layer_names:
            if(b.startswith(a)):
                _network_nodes.append(add_infix(a,b,character))
    vgg19._network_nodes = set(_network_nodes)
    
def rename(vgg19,character):        
    rename_network_nodes(vgg19,character)
    
    vgg19._name = vgg19._name + f"{character}"
    for layer in vgg19.layers:
        layer._name = layer._name + f"{character}"

######################################################################################################

########################################## Building siamese network ###################################


def build_siamese_vgg_model(shape, class_num):
    inputs1 = Input(shape)
    inputs2 = Input(shape)
    

    def one_side(x,character):
        vgg19 = VGG19(weights=None, include_top=False, input_shape=shape)
        
        rename(vgg19,character)

        x = vgg19(x)

        x = layers.Flatten()(x)
        return x
    
    x1 = one_side(inputs1,'A')
    x2 = one_side(inputs2,'B')
    

    x = layers.concatenate([x1, x2])
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)  

    # build the model
    model = Model([inputs1, inputs2], outputs)
    
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

model = build_siamese_vgg_model(x_train[0].shape, num_classes)

print(model.summary())

def make_pairs(images, labels):
    pairImages = []
    pairLabels = []

    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
#         print(currentImage.shape, label)

        
        for idxB in random.sample(list(idx[label[0]]), 1):
            posImage = images[idxB]
#             print(posImage.shape)
#             print('\n\n\n')


            pairImages.append(np.array([currentImage, posImage]))
            pairLabels.append(np.array(label))
        
    return np.array(pairImages), np.array(pairLabels)


######################################################################################################

x_train_siamese, y_train_siamese = make_pairs(x_train, Y_train)
x_test_siamese, y_test_siamese = make_pairs(x_test, Y_test)

y_train_siamese = utils.to_categorical(y_train_siamese, num_classes) 
y_test_siamese = utils.to_categorical(y_test_siamese, num_classes) 



x_train_both = [x_train_siamese[:,0,:], x_train_siamese[:,1,:]]
x_test_both = [x_test_siamese[:,0,:], x_test_siamese[:,1,:]]

print('Images')
print(np.array(x_train_both).shape)
print(np.array(x_test_both).shape)

print(model.summary())

print('Labels')
print(y_train_siamese.shape)
print(y_test_siamese.shape)


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)

history3 = model.fit(x_train_both, y_train_siamese, epochs=1, batch_size=512, shuffle=True,
         validation_data=(x_test_both, y_test_siamese),callbacks=[callback]) # id16


print('Saving model')
model.save('cifar_vgg/models/model.h5')