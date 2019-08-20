from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback,EarlyStopping
import numpy as np
import pandas as pd
import sys

LEARNING_RATE = 0.1
EPOCHS = 10000
STOP_AT_LOSS = 0.008
UNITS = 8

def scale(t):
    tMin = t.min(axis=0)
    tMax = t.max(axis=0)
    return (t-tMin)/(tMax-tMin)

class EarlyStoppingByLoss(Callback):
    def __init__(self, monitor='loss', value=STOP_AT_LOSS, verbose=0):
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
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]

            if epoch == 0:
                self.weight_dict['w_'+str(layer_i+1)] = [w]
                self.weight_dict['b_'+str(layer_i+1)] = [b]
            else:
                self.weight_dict['w_'+str(layer_i+1)].append(w)
                self.weight_dict['b_'+str(layer_i+1)].append(b)

def get_model(dim,units):
    model = Sequential()
    model.add(Dense(units,input_dim=dim,activation='tanh'))
    model.add(Dense(1,activation='sigmoid'))
    return model

def get_data():
    # Weight, height, shoe size and gender (1 = male, 0 = female)
    f = pd.read_csv("gender.csv")
    csv = f.values
    features = csv[:,:3]
    features = scale(features)
    labels = csv[:,3:]
    return (features,labels)

def main():

    (features,labels) = get_data()

    model = get_model(len(features[0]), UNITS)
    model.summary()
    model.compile(loss='binary_crossentropy',metrics=['acc'],optimizer=SGD(lr = LEARNING_RATE))

    gw = GetWeights()
    se = EarlyStoppingByLoss()

    model.fit(features, labels, batch_size=1,epochs=EPOCHS,callbacks=[gw,se],verbose=1)
    for key in gw.weight_dict:
        print((str(key) + ' shape: {}').format(np.shape(gw.weight_dict[key])))
    for key in gw.weight_dict:
        print((str(key) + ' weights: {}').format(gw.weight_dict[key][-1]))

    model.evaluate(features,labels)

if __name__ == '__main__':
        main()
