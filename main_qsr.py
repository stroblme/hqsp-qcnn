import os
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
from pennylane import numpy as np
# from scipy.io import wavfile
# import warnings
# import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.utils import plot_model
## Local Definition 
from data_generator import gen_mel
from models import cnn_Model, dense_Model, attrnn_Model
from helper_q_tool import gen_qspeech, plot_acc_loss, show_speech
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time as ti
data_ix = ti.strftime("%Y%m%d_%H%M")

labels = [
    'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
]

train_audio_path = "/storage/mstrobl/dataset/"
SAVE_PATH = "/storage/mstrobl/orig/data_quantum/" # Data saving folder

parser = argparse.ArgumentParser()
parser.add_argument("--eps", type = int, default = 30, help = "Epochs") 
parser.add_argument("--bsize", type = int, default = 16, help = "Batch Size")
parser.add_argument("--sr", type = int, default = 16000, help = "Sampling Rate for input Speech")
parser.add_argument("--net", type = int, default = 1, help = "(0) Dense Model, (1) U-Net RNN Attention")
parser.add_argument("--mel", type = int, default = 0, help = "(0) Load Demo Features, (1) Extra Mel Features")
parser.add_argument("--quanv", type = int, default = 0, help = "(0) Load Demo Features, (1) Extra Mel Features")
parser.add_argument("--port", type = int, default = 100, help = "(1/N) data ratio for encoding ")
args = parser.parse_args()

def gen_train(labels, train_audio_path, sr, port):
    all_wave, all_label = gen_mel(labels, train_audio_path, sr, port)
    return gen_train_from_wave(all_wave=all_wave, all_label=all_label)
    
def gen_train_from_wave(all_wave, all_label, output=SAVE_PATH):
    label_enconder = LabelEncoder()
    y = label_enconder.fit_transform(all_label)
    classes = list(label_enconder.classes_)
    y = keras.utils.to_categorical(y, num_classes=len(labels))

    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
    h_feat, w_feat, _ = x_train[0].shape
    np.save(output + "x_train_speech.npy", x_train)
    np.save(output + "x_test_speech.npy", x_valid)
    np.save(output + "y_train_speech.npy", y_train)
    np.save(output + "y_test_speech.npy", y_valid)
    print("===== Shape", h_feat, w_feat)

    return x_train, x_valid, y_train, y_valid

def gen_quanv(x_train, x_valid, kr, saveTo=SAVE_PATH):
    print("Kernal = ", kr)
    q_train, q_valid = gen_qspeech(x_train, x_valid, kr)

    np.save(saveTo + "quanv_train.npy", q_train)
    np.save(saveTo + "quanv_test.npy", q_valid)

    return q_train, q_valid

import sys
sys.path.append("./../")
sys.path.append("./../stqft")
from stqft.frontend import export

if __name__ == "__main__":
    if args.mel == 1:
        x_train, x_valid, y_train, y_valid = gen_train(labels, train_audio_path, args.sr, args.port) 
    else:
        x_train = np.load(SAVE_PATH + "y_train_speech.npy")
        x_valid = np.load(SAVE_PATH + "x_test_speech.npy")
        y_train = np.load(SAVE_PATH + "y_train_speech.npy")
        y_valid = np.load(SAVE_PATH + "y_test_speech.npy")

    

    # from small_quanv import gen_quanv

    if args.quanv == 1:
        q_train, q_valid = gen_quanv(x_train, x_valid, 2) 
    else:
        q_train = np.load(SAVE_PATH + "q_train_demo.npy")
        q_valid = np.load(SAVE_PATH + "q_test_demo.npy")

    ## For Quanv Exp.
    early_stop = EarlyStopping(monitor='val_loss', mode='min', 
                            verbose=1, patience=10, min_delta=0.0001)

    metric = 'val_accuracy'

    checkpoint = ModelCheckpoint('checkpoints/best_demo.hdf5', monitor=metric, 
                                verbose=1, save_best_only=True, mode='max')


    if args.net == 0:
        model = dense_Model(x_train[0], labels)
    elif args.net == 1:
        model = attrnn_Model(q_train[0], labels)

    model.summary()
    plot_model(model, to_file='model.png')

    history = model.fit(
        x=q_train, 
        y=y_train,
        epochs=args.eps, 
        callbacks=[checkpoint], 
        batch_size=args.bsize, 
        validation_data=(q_valid,y_valid)
    )


    exportPath = "/storage/mstrobl/versioning"

    model.save('checkpoints/'+ data_ix + '_demo.hdf5')
    exp = export(topic="main_qsr", identifier="model", dataDir=exportPath)
    exp.setData(export.GENERICDATA, {"history_acc":history.history['accuracy'], "history_val_acc":history.history['val_accuracy'], "history_loss":history.history['loss'], "history_val_loss":history.history['val_loss']})
    exp.setData(export.DESCRIPTION, f"Model History")
    exp.doExport()

    print("=== Batch Size: ", args.bsize)
