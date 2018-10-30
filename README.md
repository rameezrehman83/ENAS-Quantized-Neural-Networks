# Efficient Neural Architecture Search with Quantized Neural Networks

**This project combines the architecture search strategy (only micro) from [Efficient Neural Architecture Search][1] with the search space of [Quantized Neural Networks][2].** 


Introduction
------------
Efficient Neural Architecture Search recently optimized a major computational bottleneck of NAS algorithms, it does so by sharing (reusing) parameters across child models and delivers strong empirical performance. 

In ENAS, a controller discovers neural network architectures by searching for an optimal subgraph within a large computational graph. These child models are sub-graphs selected from a large computational graph which can be visualized as a directed acyclic graph. 


The controller is trained with policy gradient to select a subgraph that maximizes the expected reward on a validation set. Meanwhile, the model corresponding to the selected subgraph is trained to minimize a canonical cross entropy loss. Sharing parameters among child models allows ENAS to deliver strong empirical performances,


<p align="center">
  <img src="https://imgur.com/PO53CTS.png">
</p>


During the forward pass, Quantized Neural Networks drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations. As a result, power consumption is expected to be drastically reduced.


Project Setup 
-----
The weight sharing mechanism works by intializing the weights of the DAG only once and reusing them over various iterations, the methods used for this are `create_weight` and `create_bias` defined in [`common_ops.py`][14]. 

In the [author's code][13], they add these weights to the layers using [`tf.nn module`][9] in Tensorflow which allows the user to set custom weights to a new layer. 

As we are searching in the space of quantized neural networks, and to implement the quantization we use custom keras layers and there is no provision to set resusable weights to these layers. In the custom layer, the weights are defined using `self.add_weight` method which is defined locally in ` ./environment/lib/python3.x/site-packages/keras/engine/base_layer.py ` file the keras installation folder. 

Now, I tweaked this method slightly so that it allows to set custom weights to the layers. It is definitely not a good idea to do such changes this in your global installation of Keras, and I strongly suggest using a virtual environment for this. 

Please read this [blog][8] to know how the custom Keras layers are written, I have a [separate mini-project][15] which contains code to build these quantized networks, and perhaps it will be a good idea to take a look there before reading the code in this repository. 

I have also written a small bash script [` setup.sh `][16] that you can run and it will create the virtual environment, installing dependencies, and would take care of replacing the `base_layer.py` file in your virtual environment's directory. 

```bash 

chmod +x setup.sh 
./setup.sh 

```

Once you execute this you're ready to go! 

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

To run the architecture search, you can edit the experiment configurations in [```search_arc_cifar.py```][7] and [```search_arc_mnist.py```][9] for CIFAR10 and MNIST respectively. 

Use the following command to run the experiment finally. 

```bash 

python search_arc_cifar.py >> cifar_search.txt
python search_arc_mnist.py >> mnist_search.txt

```
All the ouput will be redirected to ``` cifar_search.txt / mnist_search.txt``` file. 


#### Analyzing Output 

In the output file, after training cycle for the controller we sample 10 architectures and valdation accuracy of these architectures. The output for the architectures will be logged as follows: 

```bash
Epoch 181: Eval
Eval at 77830
valid_accuracy: 0.9612
Eval at 77830
Test Num examples:  10000
test_accuracy: 0.9622
epoch = 181   ch_step = 77850  loss = 0.127491   lr = 0.0456   |g| = 0.2030   tr_acc = 108/128   mins = 549.07    
..   
Epoch 182: Training controller
ctrl_step = 5430   loss = 0.266   ent = 49.17   lr = 0.0035   |g| = 0.0002   acc = 0.9688   bl = 0.97   mins = 550.96
..
Here are 10 architectures
[0 2 1 4 1 3 0 1 1 0 1 0 1 2 0 4 0 0 0 1]     # Denotes the architecture for normal cell 
[1 3 1 4 0 1 1 1 1 2 1 4 3 2 0 2 1 1 0 3]     # Denotes the architecture for reduction cell 
val_acc = 0.9688
---------------------------------------------------
..
[0 0 0 1 1 0 1 3 1 1 0 3 1 1 1 0 0 4 0 0]
[1 0 1 4 1 1 1 0 1 2 0 4 0 1 4 0 0 0 0 2]
val_acc = 0.9531
---------------------------------------------------

```
The architecture with highest validation accuracy needs to be trained till convergence. The two lists printed above denote the architecture of the cell. 


#### Training Architecture  

To train an architecture till convergence pass the pass the architecture string as a parameter to  [```train_arc_mnist.py```][11] or [```train_arc_cifar.py```][12] file.

To the architecture string is just concatenation of the normal cell and reduction cell, see below:

```bash

Given Architecture: 

[0 2 1 4 1 3 0 1 1 0 1 0 1 2 0 4 0 0 0 1]     # Denotes the architecture for normal cell 
[1 3 1 4 0 1 1 1 1 2 1 4 3 2 0 2 1 1 0 3]     # Denotes the architecture for reduction cell 

The architecture string becomes: "0 2 1 4 1 3 0 1 1 0 1 0 1 2 0 4 0 0 0 1 1 3 1 4 0 1 1 1 1 2 1 4 3 2 0 2 1 1 0 3"

```

Now, to train the architecture in above example you can use the commands below: 

```bash

python train_arc_mnist.py -fixed_arc "0 2 1 4 1 3 0 1 1 0 1 0 1 2 0 4 0 0 0 1 1 3 1 4 0 1 1 1 1 2 1 4 3 2 0 2 1 1 0 3" >> mnist_arc.txt
python train_arc_cifar.py -fixed_arc "0 2 1 4 1 3 0 1 1 0 1 0 1 2 0 4 0 0 0 1 1 3 1 4 0 1 1 1 1 2 1 4 3 2 0 2 1 1 0 3" >> cifar_arc.txt
```
All the ouput will be redirected to ``` mnist_arc.txt / cifar_arc.txt``` file. 



References
----------

If you find this code useful, please consider citing the original work by the authors:

```
@article{pham2018efficient,
  title={Efficient Neural Architecture Search via Parameter Sharing},
  author={Pham, Hieu and Guan, Melody Y and Zoph, Barret and Le, Quoc V and Dean, Jeff},
  journal={arXiv preprint arXiv:1802.03268},
  year={2018}
}
```

```
@article{Hubara2017QuantizedNN,
  title={Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations},
  author={Itay Hubara and Matthieu Courbariaux and Daniel Soudry and Ran El-Yaniv and Yoshua Bengio},
  journal={Journal of Machine Learning Research},
  year={2017},
  volume={18},
  pages={187:1-187:30}
}
```

Thanks to 
---------

This work wouldn't have been possible without the help from the following repos:

1. https://github.com/melodyguan/enas (Author's code)
2. https://github.com/DingKe/nn_playground/
3. https://github.com/MINGUKKANG/ENAS-Tensorflow

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
[11]:https://github.com/yashkant/ENAS-Quantized-Neural-Networks/blob/master/train_arc_cifar.py
[12]:https://github.com/yashkant/ENAS-Quantized-Neural-Networks/blob/master/train_arc_mnist.py
[13]:https://github.com/MINGUKKANG/ENAS-Tensorflow
[14]:https://github.com/yashkant/ENAS-Quantized-Neural-Networks/blob/master/enas/common_ops.py
[15]:https://github.com/yashkant/Quantized-Nets
[16]:https://github.com/yashkant/ENAS-Quantized-Neural-Networks/blob/master/enas/setup.sh
