#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:17:21 2021

@author: matsumotoyuuta
"""
from __future__ import print_function
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib
import qtt
from qtt.algorithms.random_telegraph_signal import generate_RTS_signal
from qtt.algorithms.markov_chain import ContinuousTimeMarkovModel, generate_traces
import itertools
import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from scipy import stats
import time
from tensorflow.keras.callbacks import LearningRateScheduler

def step_decay(epoch):
    x = 0.003
    if epoch >= 200: x = 0.001
    if epoch >= 800: x = 0.0001
    return x
lr_decay = LearningRateScheduler(step_decay)

class DNN_classifier():
    def __init__(self,readout_type='ES',cnn_filter=25,training_type='Exp'):
        self.readout_type = readout_type
        self.cnn_filter = cnn_filter
        self.training_type = training_type
    def build_classifier(self,input_size=480,output_size=2,cnn_filter=25,model_type=1):
        ''' 
        Build the readout classifier depending on the readout scheme.
        Currently support Energy selective readout and PSB readout
        ES type 1 : Stable training
        ES type 2 : Faster computational time than type 1, Less training stability (overfitting)
        ES type 3 : For FPGA implementation using 'hls4ml'
        PSB type 1 : Stable training
        PSB type 2 : Faster computational time than type 1, Less training stability (overfitting)
        '''
        
        if self.readout_type == 'ES':
            if model_type ==1:
                self.model = Sequential()
                self.model.add(Conv1D(1, cnn_filter, input_shape=(input_size,1)))
                self.model.add(LeakyReLU(alpha =0.0))
                self.model.add(Conv1D(1, cnn_filter))
                self.model.add(LeakyReLU(alpha =0.0))
                self.model.add(Conv1D(1, cnn_filter))    
                
                if output_size ==2:
                    self.model.add(LSTM(output_size,activation='sigmoid'))# Same as softmax, better for stable training.
                else:
                    self.model.add(LSTM(output_size,activation='softmax'))
                print(self.model.summary())
                
            elif model_type ==2:
                self.model = Sequential()
                self.model.add(Conv1D(1, cnn_filter, input_shape=(input_size,1)))
                self.model.add(BatchNormalization())
                self.model.add(LeakyReLU(alpha =0.0))
                self.model.add(Conv1D(1, cnn_filter))
                self.model.add(BatchNormalization())
                self.model.add(LeakyReLU(alpha =0.0))
                self.model.add(Conv1D(1, cnn_filter))
                self.model.add(BatchNormalization())
                self.model.add(LeakyReLU(alpha =0.0))
                self.model.add(Flatten())
                self.model.add(Dense(200))
                self.model.add(Dense(200))
                if output_size ==2:
                    self.model.add(Dense(output_size,activation='sigmoid')) 
                else:
                    self.model.add(Dense(output_size,activation='softmax')) 
            elif model_type ==3:
                self.model = Sequential()
                self.model.add(Flatten(input_shape=(input_size,1)))
                self.model.add(Dense(32))
                self.model.add(BatchNormalization())
                self.model.add(Dense(16))
                self.model.add(BatchNormalization())
                self.model.add(Dense(16))
                if output_size ==2:
                    self.model.add(Dense(output_size,activation='sigmoid')) 
                else:
                    self.model.add(Dense(output_size,activation='softmax')) 
                print(self.model.summary())
                    
        if self.readout_type == 'PSB':
            if model_type ==1:
                self.model = Sequential()
                self.model.add(Conv1D(1, cnn_filter, input_shape=(input_size,1)))
                self.model.add(BatchNormalization())
                self.model.add(LeakyReLU(alpha =0.0))
                self.model.add(Conv1D(1, cnn_filter))
                self.model.add(BatchNormalization())
                self.model.add(LeakyReLU(alpha =0.0))
                self.model.add(Conv1D(1, cnn_filter))
                self.model.add(BatchNormalization())
                self.model.add(LeakyReLU(alpha =0.0))
                self.model.add(Flatten())
                self.model.add(Dense(32))
                self.model.add(Dense(16))
                if output_size ==2:
                        self.model.add(Dense(output_size,activation='sigmoid')) 
                else:
                    self.model.add(Dense(output_size,activation='softmax')) 
                print(self.model.summary())
            elif model_type ==2:
                self.model = Sequential()
                self.model.add(Flatten(input_shape=(input_size,1)))
                self.model.add(Dense(32))
                self.model.add(Dense(32))
                self.model.add(Dense(16))
                if output_size ==2:
                        self.model.add(Dense(output_size,activation='sigmoid')) 
                else:
                    self.model.add(Dense(output_size,activation='softmax')) 
                print(self.model.summary())

    def train(self,nepochs=100,batch_size=64):
            self.model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
            history = self.model.fit(self.training_dataset[0], self.training_dataset[1],
                                  batch_size=batch_size,
                                  epochs=nepochs,
                                  validation_split=0,
                                    callbacks=[lr_decay],
                                  verbose=1)

    def make_training_dataset(self,readout_traces,short_length=50,long_length=400,data_aug= False):
        ''' 
        Correct training dataset in experiment. 
        For energy selective readout, we obtain the signal on charge transition line near readout point and coulomb blockade region (Background noise)
        For PSB readout scheme, We assign the label using long integration time(See Note). We assume T1 from T to S state is long enough. 
        You can do so, for example, with Latching PSB and integration on an appropriate energy detuning.
        
        readout_traces: (For ES) Traces obtained on charge transition line and in a coulomb blockade region 
        readout_traces: (For PSB) Traces of readout stage, that should includes S and T state equally (roughly 40~60%). 
        '''
        state_labels = np.empty(0)
        if self.readout_type == 'ES':
            state_labels = np.append(state_labels,np.ones(len(readout_traces)/2))
            state_labels = np.append(state_labels,np.zeros(len(readout_traces)/2))
            if data_aug == True:
                pass
            self.training_dataset = list(readout_traces,state_labels) 
        if self.readout_type == 'PSB':
            readout_traces = readout_traces[:,:short_length] # Input for DNNs
            state_labels = 1*(np.mean(readout_traces[:,:long_length],axis=1)>threshold) # assign labels (Output for DNNs) using integrated readout value 
            self.training_dataset = list(readout_traces,state_labels) 
            
            
    def load_pretrained_model(self,pre_model):
        ''' 
        Load pretrained model from the model repository
        This function is used for the option of the fine tuning. 
        We found that training the pre-trained model with experimental dataset shows robust and fast training.
        '''
        self.model = keras.models.load_model(pre_model)
            
class ES_data_augmenter():       
    ''' 
    This class is designed for data augmentation of the training dataset.
    It is useful to correctly generalize the model to detect the blip signals.
    If you face overfitting preblems in the training precedure, we would recommend to use it. 
    '''
    def __init__(self,readoutlength,exp_sig = None,offset_variation=False,mix_simdata = True,exp_noise= None,noise=0.5,signal_amp=1,tunnel_time=None,num_samples=20000):
        if mix_simdata == True:
            t_rate=1/(readoutlength*1e-6/12)
            readoutlength = readoutlength
            noise=noise
            # model_unit = 1e-6 # we work with microseconds as the base unit
            # rate_up = t_rate
            # rate_down = t_rate
            # rts_model = ContinuousTimeMarkovModel(['zero', 'one'], [rate_up*model_unit,rate_down*model_unit], np.array([[0.,1],[1,0]]))
            # rts_data = generate_traces(rts_model, number_of_sequences=40000, length=readoutlength, std_gaussian_noise=0, delta_time=1)
            rts_data = np.zeros((num_samples,readoutlength))
            rts_data2 = np.zeros((num_samples,readoutlength))
            xdataset=[]
            xdataset2=[]
            ydataset=[]
            xdataset.append(rts_data)
            xdataset.append(rts_data2)
            ydataset.append(np.ones(num_samples))
            ydataset.append(np.zeros(num_samples))
            ydataset = np.array(ydataset)
            xdataset=np.array(xdataset).reshape(num_samples*2,readoutlength)
            if tunnel_time is None:
                tunnel_time = readoutlength/6
            for i in range(num_samples):
                blip = np.random.normal(tunnel_time,tunnel_time/2)
                blip_start = random.randrange(2,int(readoutlength-blip+2),1)-2
                try :
                    xdataset[i,int(blip_start):int(blip_start+blip)]=1
                except IndexError:
                    pass
                
            # for i in range(40000):
            #     try :
            #         xdataset[i,np.where(xdataset[i,np.where(xdataset[i]>=1)[0][0]:]==0)[0][0]+np.where(xdataset[i]>=1)[0][0]:]=0
            #     except IndexError:
            #         pass
                
            if exp_noise is None:
                xdataset = (xdataset+np.random.normal(0, noise, (len(xdataset),readoutlength)))*signal_amp
            else:
                if len(exp_noise)<num_samples*2:
                    reps=int(num_samples/len(exp_noise))
                    for k in range(reps):
                        exp_noise = np.append(exp_noise,exp_noise,axis=0)
                xdataset = (xdataset+noise*scipy.stats.zscore(exp_noise)[:len(xdataset),:readoutlength])*signal_amp
        if exp_sig is not None:
            xdataset = np.append(xdataset,exp_sig,axis=0)
            ydataset = np.append(ydataset,np.ones(len(exp_sig)))
        if offset_variation :
            xdataset2 = np.empty(0)
            ydataset2 = np.empty(0)
            for i in range(5):
                xdataset2 = np.append(xdataset2,xdataset+np.random.normal(0, 0.1*signal_amp))
                ydataset2 = np.append(ydataset2,ydataset)
            xdataset = xdataset2
            ydataset = ydataset2
        xdataset=np.array(xdataset).reshape(len(xdataset),readoutlength,1)
        ydataset=np.array(ydataset).reshape(len(xdataset))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(xdataset, ydataset, test_size=0.05, random_state=0)
            
            
        