# Federated Learning with Dropout

## Introduction 
This repository contains implementations of various federated learning methods, including FedProx, ReBAFL, RFA, and FedAR, with a focus on dropout techniques. The experiments are conducted in both IID and non-IID settings using 2D data.


The experiments utilize DenseNet architectures as baseline models and extend to 2D data applications. By implementing and comparing these diverse approaches, the project aims to identify optimal combinations of federated learning algorithms and dropout techniques for various real-world scenarios where data is distributed across multiple clients

## Project Structure 
```
federated-learning-dropout/
│
├── README.md
│
├── methods/
│   ├── fedprox2d.ipynb
│   ├── rebafl.ipynb
│   ├── RFA.ipynb
│   └── FedAR/
│       ├── FedAR-iid-2d.ipynb
│       └── fedar2d.ipynb
│
└── dropout/
    ├── iid-exp.ipynb
    ├── non-iid-exp.ipynb
    └── baseline/
        ├── FL_DenseNet.ipynb
        └── FL_DenseNet2D.ipynb
```
