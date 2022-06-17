import numpy as np
np.random.seed(1234567)
import random
random.seed(1234567)

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from tensorflow.python import debug as tf_debug

from pymatgen import Composition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.utils.conversions import str_to_composition

from collections import Counter
import re, math, operator, sys, argparse, time
import joblib

SEED = 1234567
elements = ['H','Li','Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl','K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge','As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd','Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd','Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']
featurizer = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"), cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])

# Regex to Choose from Element Name, Number and Either of the Brackets
token = re.compile('[A-Z][a-z]?|\d+|[()]')

def toList(string): 
    li = list(string.split(" ")) 
    return li

# Create a dictionary with the Name of the Element as Key and No. of elements as Value
def count_elements(formula):
    tokens = token.findall(formula)
    stack = [[]]
    for t in tokens:
        if t.isalpha():
            last = [t]
            stack[-1].append(t)
        elif t.isdigit():
             stack[-1].extend(last*(int(t)-1))
        elif t == '(':
            stack.append([])
        elif t == ')':
            last = stack.pop()
            stack[-1].extend(last)   
    return dict(Counter(stack[-1]))

#Normalize the Value of the Dictionary
def normalize_elements(dictionary):
    factor=1.0/sum(dictionary.values())
    for k in dictionary:
        dictionary[k] = dictionary[k]*factor
    return dictionary

def input_elements(compounds):
    in_elements = np.zeros(shape=(len(compounds), len(elements)))
    comp_no = 0

    for compound in compounds:
        keys = compound.keys()
        for key in keys:
            in_elements[comp_no][elements.index(key)] = compound[key]
        comp_no+=1  
    data = in_elements   
    return data 

def compound_to_ef(compounds):
    compound = [count_elements(x) for x in compounds]
    compound = [normalize_elements(x) for x in compound]
    compound = input_elements(compound)

    return compound

def compound_to_pa(compounds):
    compound_obj = [Composition(compound) for compound in compounds]
    compound_pa = featurizer.featurize_many(compound_obj, ignore_errors=True)
    compound_pa = np.asarray(compound_pa)

    return compound_pa

def define_model(data, architecture, num_labels=1, activation='relu', dropouts=[]):

        assert '-' in architecture
        archs = architecture.strip().split('-')
        net = data
        pen_layer = net
        prev_layer = net
        prev_num_outputs = None
        prev_block_num_outputs = None
        prev_stub_output = net
        for i in range(len(archs)):
            arch = archs[i]
            if 'x' in arch:
                arch = arch.split('x')
                num_outputs = int(re.findall(r'\d+',arch[0])[0])
                layers = int(re.findall(r'\d+',arch[1])[0])
                j = 0
                aux_layers = re.findall(r'[A-Z]',arch[0])
                for l in range(layers):
                    if aux_layers and aux_layers[0] == 'B':
                        if len(aux_layers)>1 and aux_layers[1]=='A':
                            #print('adding fully connected layers with %d outputs followed by batch_norm and act' % num_outputs)

                            net = Dense(num_outputs, 
                                        name='fc' + str(i) + '_' + str(j),
                                        activation=None)(net)
                            net = BatchNormalization(center=True, scale=True, name='fc_bn'+str(i)+'_'+str(j))(net)
                            if activation =='relu': net = Activation('relu')(net)
                        else:
                            #print('adding fully connected layers with %d outputs followed by batch_norm' % num_outputs)
                            net = Dense(num_outputs,
                                        name='fc' + str(i) + '_' + str(j),
                                        activation=activation)(net)
                            net = BatchNormalization(center=True, scale=True,
                                             name='fc_bn' + str(i) + '_' + str(j))(net)

                    else:
                        #print('adding fully connected layers with %d outputs' % num_outputs)

                        net = Dense(num_outputs,
                                    name='fc' + str(i) + '_' + str(j), 
                                    activation=activation)(net)

                    if 'R' in aux_layers:
                        if prev_num_outputs and prev_num_outputs==num_outputs:
                            #print('adding residual, both sizes are same')

                            net = net+prev_layer
                        else:
                            #print('adding residual with fc as the size are different')
                            net = net + Dense(num_outputs,
                                                name='fc' + str(i) + '_' +'dim_'+ str(j),
                                                activation=None)(prev_layer)
                    prev_num_outputs = num_outputs
                    j += 1
                    prev_layer = net
                aux_layers_sub = re.findall(r'[A-Z]', arch[1])
                if 'R' in aux_layers_sub:
                    if prev_block_num_outputs and prev_block_num_outputs == num_outputs:
                        #print('adding residual to stub, both sizes are same')
                        net = net + prev_stub_output
                    else:
                        #print('adding residual to stub with fc as the size are different')
                        net = net + Dense(num_outputs,
                                         name='fc' + str(i) + '_' + 'stub_dim_' + str(j),
                                         activation=None)(prev_stub_output)

                if 'D' in aux_layers_sub and (num_labels == 1) and len(dropouts) > i:
                    #print('adding dropout', dropouts[i])
                    net = Dropout(1.-dropouts[i], seed=SEED)(net, training=False)
                prev_stub_output = net
                prev_block_num_outputs = num_outputs
                prev_layer = net

            else:
                if 'R' in arch:
                    act_fun = 'relu'
                    #print('using ReLU at last layer')  
                else:
                    act_fun = None
                pen_layer = net
                #print('adding final layer with ' + str(num_labels) + ' output')
                net = Dense(num_labels, name='fc' + str(i),
                            activation=act_fun)(net)

        return net

def model_prediction(model, model_path, data):
    model.load_weights(model_path)
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])
    predict = model.predict(data)

    return predict

def model_prediction_fe(model, target_model, model_path, target_model_path, data, layer_no):
    model.load_weights(model_path)
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])
    
    extractor = tf.keras.models.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
    features = extractor(data)
    feature = features[layer_no].numpy()
    
    target_model.load_weights(target_model_path)
    adam = optimizers.Adam(lr=0.0001)
    target_model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])
    predict = target_model.predict(feature)    

    return predict

def ml_model_prediction(target_model_path, data):
    loaded_model = joblib.load(target_model_path)
    predict = loaded_model.predict(data) 

    return predict

def ml_model_prediction_fe(model, model_path, target_model_path, data, layer_no):
    model.load_weights(model_path)
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])
    extractor = tf.keras.models.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
    features = extractor(data)
    feature = features[layer_no].numpy()

    loaded_model = joblib.load(target_model_path)
    predict = loaded_model.predict(feature) 

    return predict


       
