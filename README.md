# Hybrid Quantum Speech Processing - QCNN repository

This is the qcnn directory of the *hqsp* project.

The initial code used within this repository is copied from the Github repository [QuantumSpeech-QCNN](https://github.com/huckiyang/QuantumSpeech-QCNN) which is part of the work of [Decentralizing Feature Extraction with Quantum Convolutional Neural Network for Automatic Speech Recognition](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1109%2FICASSP39728.2021.9413453&v=3eab440e).

In context of the "Hybrid Quantum Speech Processing" project, changes to the model architecture in [models.py](models.py) were made.
The files [small_qsr.py](small_qsr.py) and [small_quanv.py](small_quanv.py) were added to enable multiprocessing calls and to introduce layers of abstraction but are mainly build on the ideas of aforementioned authors.

Relevant files for the work in this project are therefore:

- [data_generator.py](data_generator.py)
- [models.py](models.py)
- [small_qsr.py](small_qsr.py)
- [small_quanv.py](small_quanv.py)

License file from original repository was carried over.

## Usage

Files in this repository are not intended for direct usage.

Quantum speech processing and Pixel-Channel-Mapping (noQuanv): [small_quanv.py](small_quanv.py)
Classical spectrogram generation abstract., labels and set splitting: [small_qsr.py](small_qsr.py)
Neural network model, custom derivative: [models.py](models.py)
Classical spectrogram generation: [data_generator.py](data_generator.py)