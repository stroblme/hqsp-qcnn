import os

## Local Definition 
from data_generator import gen_mel
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64/:/usr/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-11.2/lib64:/usr/local/cuda/targets/x86_64-linux/lib/"

import time as ti
data_ix = ti.strftime("%Y%m%d_%H%M")

labels = [
    'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
]

train_audio_path = "/ceph/mstrobl/dataset/"

def gen_train(labels, train_audio_path, sr, port):
    all_wave, all_label = gen_mel(labels, train_audio_path, sr, port)
    return gen_train_from_wave(all_wave=all_wave, all_label=all_label)
    
def gen_train_from_wave(all_wave, all_label, output):
    import numpy as np
    from tensorflow import keras

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    label_enconder = LabelEncoder()
    y = label_enconder.fit_transform(all_label)
    classes = list(label_enconder.classes_)
    y = keras.utils.to_categorical(y, num_classes=len(labels))

    x_train, x_valid, y_train, y_valid = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
    h_feat, w_feat, _ = x_train[0].shape
    np.save(output + "x_train_speech.npy", x_train)
    np.save(output + "x_test_speech.npy", x_valid)
    np.save(output + "y_train_speech.npy", y_train)
    np.save(output + "y_test_speech.npy", y_valid)
    print("===== Shape", h_feat, w_feat)

    return x_train, x_valid, y_train, y_valid

def gen_train_from_wave_no_split(all_wave, all_label):
    import numpy as np
    from tensorflow import keras

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    label_enconder = LabelEncoder()
    y = label_enconder.fit_transform(all_label)
    classes = list(label_enconder.classes_)
    y = keras.utils.to_categorical(y, num_classes=len(labels))

    return np.array(all_wave), np.array(y)