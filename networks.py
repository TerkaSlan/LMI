import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.regularizers import l2

from imports import logging
logging.basicConfig(datefmt='%d-%m-%y %H:%M', format='%(asctime)-15s%(levelname)s: %(message)s', level=logging.INFO)


def compile(model, loss='sparse_categorical_crossentropy', metrics=['accuracy'], opt=Adam(learning_rate=0.001)):
    model.compile(loss=loss, metrics=metrics, optimizer=opt)
    return model

def construct_conv1d_model(input_data_shape=282, output_data_shape=16):
    model = Sequential(name="conv1d")
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_data_shape, 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_data_shape, activation='softmax'))
    
    return compile(model, 'categorical_crossentropy', metrics=["categorical_accuracy"])

def construct_mlp(input_data_shape=282, output_data_shape=16):
    reg1 = l2(0.0001)
    model = Sequential(name="MLP")
    model.add(Dense(units=100, activation='relu', input_dim=input_data_shape,  kernel_regularizer=reg1, bias_regularizer=reg1, activity_regularizer=reg1))
    model.add(Dense(units=output_data_shape, kernel_initializer='glorot_uniform', activation='softmax'))
    
    return compile(model)

def construct_fully_connected_model(input_data_shape=282, output_data_shape=16):
    logging.info("[32]-[32] model")
    model = Sequential(name="simple_fully_conn")
    model.add(Dense(units=32, activation='relu', input_dim=input_data_shape))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=output_data_shape, activation='softmax'))

    return compile(model)

def construct_fully_connected_model_282_128(input_data_shape=282, output_data_shape=16):
    logging.info("[282]-[128] model")
    model = Sequential(name="simple_fully_conn")
    model.add(Dense(units=282, activation='relu', input_dim=input_data_shape))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=output_data_shape, activation='softmax'))

    return compile(model)

def construct_baseline_model(input_data_shape=282, output_data_shape=18):
    logging.info("[282]-[1024]-[256] multilabel model")
    clf = Sequential(name="multilabel_baseline")
    clf.add(Dense(input_data_shape, activation='relu', input_dim=input_data_shape))
    clf.add(Dense(units=1024, activation='relu'))
    clf.add(Dense(units=256, activation='relu'))
    clf.add(Dense(output_data_shape, activation='sigmoid'))
    
    return compile(clf)

def construct_simple_baseline_model(input_data_shape=282, output_data_shape=18):
    logging.info("[282]- multilabel model")
    clf = Sequential(name="multilabel_simple_model")
    clf.add(Dense(input_data_shape, activation='relu', input_dim=input_data_shape))
    clf.add(Dense(output_data_shape, activation='sigmoid'))
    
    return compile(clf)
