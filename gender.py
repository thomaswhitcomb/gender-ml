#from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback,EarlyStopping
import numpy as np
import pandas as pd
import sys

def scale(t):
    tMin = t.min(axis=0)
    tMax = t.max(axis=0)
    return (t-tMin)/(tMax-tMin)

class EarlyStoppingByLoss(Callback):
    def __init__(self, monitor='loss', value=0.008, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}):
        global lastEpoch
        current = logs.get("loss")         
        if current != None and current < self.value:
            self.model.stop_training = True
            lastEpoch = epoch + 1

class GetWeights(Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]
            #print('Layer %s has weights of shape %s and biases of shape %s' %(layer_i, np.shape(w), np.shape(b)))

            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                self.weight_dict['w_'+str(layer_i+1)] = [w]
                self.weight_dict['b_'+str(layer_i+1)] = [b]
            else:
                # append new weights to previously-created weights array
                self.weight_dict['w_'+str(layer_i+1)].append(w)
                # append new weights to previously-created weights array
                self.weight_dict['b_'+str(layer_i+1)].append(b)

# Weight, height, shoe size and gender (1 = male, 0 = female)
f = pd.read_csv("gender.csv")
csv = f.values
data = csv[:,:3]
data = scale(data)
labels = csv[:,3:]

model = Sequential()
model.add(Dense(8,input_dim=3,activation='tanh'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',metrics=['acc'],optimizer=SGD(lr = 0.1))

model.summary()


# Train the model, iterating on the data in batches of 32 samples
gw = GetWeights()
se = EarlyStoppingByLoss()

history = model.fit(data, labels, batch_size=1,epochs=10000,callbacks=[gw,se],verbose=1)
for key in gw.weight_dict:
    print((str(key) + ' shape: {}').format(np.shape(gw.weight_dict[key])))
for key in gw.weight_dict:
    print((str(key) + ' weights: {}').format(gw.weight_dict[key][-1]))
#model.evaluate(data,labels)
