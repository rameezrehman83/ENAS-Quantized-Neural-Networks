# Efficient Neural Architecture Search with Quantized Neural Networks

**This project combines the architecture search strategy from [Efficient Neural Architecture Search][1] with the search space of [Quantized Neural Networks][2].** 



Introduction
------------
Neural Architecture Search is a sub-field of AutoML, which garnered popularity after Neural Architecture Search with RL showed promising results. 

ENAS shares parameters across child models that allows for strong empirical performance, and delivers strong empirical performance. 

The bottleneck of using NAS remains the computational resources needed for it, ENAS provides a very efficient way to cope up with these drawbacks by reusing parameters across child models by sharing them. 

These child models are sub-graphs selected from a large computational graph which can be visualized as a directed acyclic graph. 

In ENAS, a controller discovers neural network architectures by searching for an optimal subgraph within a large computational graph. 

The controller is trained with policy gradient to select a subgraph that maximizes the expected reward on a validation set. 

Meanwhile the model corresponding to the selected subgraph is trained to minimize a canonical cross entropy loss. Sharing parameters among child models allows ENAS to deliver strong empirical performances,


<p align="center">
  <img src="https://imgur.com/u5ALF0u.png">
</p>


Binarized Neural Networks with binary weights and activations at run-time drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations which substantially improve power-efficiency. Both the weights and the activations are constrained to either +1 or -1. 

Binarization function used in the experiment is deterministic binary-tanh which is placed in [```binary_ops.py```][3]. 


Project Setup 
-----
The weight sharing mechanism works by intializing the weights of the DAG only once and reusing them over various iterations, the methods used for this are `create_weight` and `create_bias` defined in `common_ops.py` (Add Link).

In the author's code (add link), they add these weights to the layers using [tf.nn module][9] in Tensorflow which allows the user to set custom weights to a new layer. 

Now, we are searching in the space of quantized neural networks, and to implement the quantization we use custom keras layers and there is no provision to set resusable weights to these layers. In the custom layer, the weights are defined using `self.add_weight` method which is defined locally in --- file the keras installation folder. 

Now, I tweaked this method slightly so that it allows to set custom weights to the layers. It is definitely not a good idea to tweak this in your global installation of Keras, and I strongly suggest using a virtual environment for this. 

Please read this [blog][8] to know how the custom Keras layers are written, I have separate mini-project which contains code to build these quantized networks, and perhaps it will be a good idea to take a look there, before reading the code in this repository. 




Project Structure
-----------------
The skeletal overview of the project is as follows: 

```bash

├── binarize/
│   ├── binary_layers.py  # Custom binary layers are defined in Keras 
│   └── binary_ops.py     # Binarization functions for weights and activations
|
├── ternarize/
│   ├── ternary_layers.py  # Custom ternarized layers are defined in Keras
│   └── ternary_ops.py     # Ternarization functions for weights and activations
|
├── quantize/
│   ├── quantized_layers.py  # Custom quantized layers are defined in Keras
│   └── quantized_ops.py     # Quantization functions for weights and activations
|
├── enas/                               
│   ├── data_utils.py & data_utils_cifar.py  # Code to pre-process and import datasets
│   ├── micro_controller.py                  # Builds the controller graph 
│   ├── common_ops.py                        # Contain methods needed for reusing weights
│   ├── models.py & controller.py            # Base classes for MicroChild and MicroController
│   ├── utils.py                             # Methods to build training operations graph
│   └── micro_child.py                       # Builds the graph for child model from the architecture string
|
├── main_controller_child_trainer.py         # Defines experiment settings and runs architecture search        
└── main_child_trainer.py                    # Trains given architecture till convergence  
```


Defining Experiment Configuration 
---------------------------------

#### Datasets

Extract the three zip files stored in [``` data/mnist ```][6] in the same folder for the MNIST experiment, for the cifar10 experiment read the directions in file [```cifar10_dataset.txt```][3]. 


#### Architecture Search

To run the architecture search experiment, you can edit the configurations in [```search_arc_cifar.py```][7] or [```search_arc_mnist.py```][9] for CIFAR10 and MNIST respectively. 

Use the following command to run the experiment finally. 

```bash 

python search_arc_cifar.py >> cifar_search.txt
python search_arc_mnist.py >> mnist_search.txt

```
All the ouput will be redirected to ``` cifar_search.txt ``` file. 


#### Analyzing Output 

All the trained architectures are stored in ```architectures/{EXPERIMENT_NAME}.txt ``` file. The output for an architecture will be logged as follows: 

```bash
Sr. No: 1
Reward: 0.4846  # Defines the reward/accuracy 
Architecture: [0, '3x3 sep-bconv', 0, '3x3 sep-bconv']  # Architecture Specification 
Representation String: "[[1. 0. 0.]] [[1. 0. 0. 0.]] [[1. 0. 0.]] [[1. 0. 0. 0.]]"  # This will be used for training architectures till convergence
```
The architecture with highest reward needs to be trained till convergence, follow the steps below for it. 


#### Training Architecture  

To train an architecture till convergence edit the following section of [```train.py```][7] file. Pick the required architecture's representation string (see above) from the output and replace the corresponding field below with it. 

```bash

# -------Architecture Training Settings-----
NUM_EPOCHS = 200  # Define the number of epochs.
REPRESENTATION_STRING = "[[1. 0. 0.]] [[1. 0. 0. 0.]] [[1. 0. 0.]] [[1. 0. 0. 0.]]"  # Replace this string with the architecture representation string required
LOAD_SAVED = False # Set this to true to continue training a saved architecture 
# ------------------------------------------

```
After replacing the ```REPRESENTATION_STRING``` run the following command:

```bash

python train.py -ta True

```


References
----------

If you find this code useful, please consider citing the original work by the authors:

```
@article{liu2017progressive,
  title={Progressive neural architecture search},
  author={Liu, Chenxi and Zoph, Barret and Shlens, Jonathon and Hua, Wei and Li, Li-Jia and Fei-Fei, Li and Yuille, Alan and Huang, Jonathan and Murphy, Kevin},
  journal={arXiv preprint arXiv:1712.00559},
  year={2017}
}
```

```
@inproceedings{hubara2016binarized,
  title={Binarized neural networks},
  author={Hubara, Itay and Courbariaux, Matthieu and Soudry, Daniel and El-Yaniv, Ran and Bengio, Yoshua},
  booktitle={Advances in neural information processing systems},
  pages={4107--4115},
  year={2016}
}
```




[1]:https://arxiv.org/abs/1802.03268
[2]:https://arxiv.org/abs/1609.07061
[3]:https://github.com/yashkant/ENAS-Quantized-Neural-Networks/blob/master/data/cifar10_dataset.txt
[4]:https://www.tensorflow.org/install/
[5]:https://keras.io/#installation
[6]:https://github.com/yashkant/ENAS-Quantized-Neural-Networks/tree/master/data/mnist
[7]:https://github.com/yashkant/ENAS-Quantized-Neural-Networks/blob/master/search_arc_cifar.py
[8]:https://keras.io/layers/writing-your-own-keras-layers/
[9]:https://www.tensorflow.org/api_docs/python/tf/nn
[10]:https://github.com/yashkant/ENAS-Quantized-Neural-Networks/blob/master/search_arc_mnist.py


Thanks to 
---------

This work wouldn't have been possible without the help from the following repos:

1. https://github.com/titu1994/progressive-neural-architecture-search
2. https://github.com/DingKe/nn_playground/
