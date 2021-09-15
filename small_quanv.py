import pennylane as qml
import numpy as np
from pennylane.templates import RandomLayers

import multiprocessing
# from multiprocessing import Pool
from multiprocessing import Pool


#TODO: investigate in those hyperparameters
n_layers = 1
n_w = 4 # numbers of wires def 4
noise_mode = False # for running at QPU

if  noise_mode == True:
    dev = qml.device('qiskit.aer', wires= n_w, noise_model=noise_model)
else:
    dev = qml.device("default.qubit", wires= n_w)


#TODO: maybe move that to the corresponding fct?
# Random circuit parameters
rand_params = np.random.uniform(high= 2 * np.pi, size=(n_layers, n_w)) # def 2, n_w = 4

kernelSize = 2 # will be overwritten in gen_qspeech

@qml.qnode(dev)
def circuit(phi=None):
    # Encoding of 4 classical input values
    for j in range(n_w):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(n_w)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(n_w)]

def quanv(image, kr=2):
    h_feat, w_feat, ch_n = image.shape
    """Convolves the input speech with many applications of the same quantum circuit."""
    out = np.zeros((h_feat//kr, w_feat//kr, n_w))
    # print(f"{image.min()}, {image.dfamax()}")
    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, h_feat, kr):
        for k in range(0, w_feat, kr):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                # kernal 3 ## phi=[image[j, k, 0], image[j, k + 1, 0], image[j, k + 2, 0], image[j + 1, k, 0], 
                # image[j + 1, k + 1, 0], image[j + 1, k +2 , 0],image[j+2, k, 0], image[j+2, k+1, 0], image[j+2, k+2, 0]]
                phi=[image[j, k, 0], image[j, k + 1, 0], image[j + 1, k, 0], image[j + 1, k + 1, 0]]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(n_w):
                out[j // kr, k // kr, c] = q_results[c]
    return out

def poolQuanv(img):
    return quanv(img, kernelSize)

def gen_qspeech(x_train, x_valid, kr, poolSize=1): # kernal size = 2x2 or 3x3
    global kernelSize
    kernelSize = kr # moving the local variable to global here for pool processing

    q_train = list()
    temp_q = list()
    print("Quantum pre-processing of train Speech:")
    
    x_train = x_train * (1/x_train.max())

    with Pool(poolSize) as p:
        q_train = p.map(poolQuanv, x_train)
        
    q_train = np.asarray(q_train)

    if x_valid == []:
        print("Validation array is empty! Ensure that we are running test sets!")
        return q_train
        
    q_valid = list()
    print("\nQuantum pre-processing of valid Speech:")
    
    x_valid = x_valid * (1/x_valid.max())

    with Pool(poolSize) as p:
        q_valid = p.map(poolQuanv, x_valid)

    q_valid = np.asarray(q_valid)
    
    return q_train, q_valid

def gen_quanv(x_train, x_valid, kr, output, poolSize=1):
    print("Kernal = ", kr)
    q_train, q_valid = gen_qspeech(x_train, x_valid, kr, poolSize)

    np.save(f"{output}/quanv_train.npy", q_train)
    np.save(f"{output}/quanv_valid.npy", q_valid)

    return q_train, q_valid