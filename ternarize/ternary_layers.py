# -*- coding: utf-8 -*-
import numpy as np

from keras import backend as K
import tensorflow as tf
from keras.layers import InputSpec, Dense, Conv2D, SimpleRNN, SeparableConv2D
from keras import constraints
from keras import initializers

from ternarize.ternary_ops import ternarize

class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value,
                "max_value": self.max_value}

class TernaryDense(Dense):
    ''' Ternarized Dense layer

    References: 
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902}
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    '''
    def __init__(self, units, H=1., kernel_lr_multiplier='Glorot', bias_lr_multiplier=None, ternarize = True, w_getter = None, **kwargs):
        super(TernaryDense, self).__init__(units, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        self.ternarize = ternarize
        self.w_getter = w_getter
        
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot H: {}'.format(self.H))
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot learning rate multiplier: {}'.format(self.kernel_lr_multiplier))
            
        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint, set_weight = self.w_getter)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.output_dim,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        if(self.ternarize):
            # print("------Ternarize Weights------")
            ternary_kernel = ternarize(self.kernel, H=self.H) 
        else:
            # print("------ No Ternarize Weights------")
            ternary_kernel = self.kernel
        output = K.dot(inputs, ternary_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
        
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(TernaryDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TernaryConv2D(Conv2D):
    '''Ternarized Convolution2D layer
    References: 
    - [Recurrent Neural Networks with Limited Numerical Precision](http://arxiv.org/abs/1608.06902}
    - [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
    '''
    def __init__(self, filters, kernel_lr_multiplier='Glorot', 
                 bias_lr_multiplier=None, H=1., ternarize = True, w_getter = None, **kwargs):
        super(TernaryConv2D, self).__init__(filters, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        self.ternarize = ternarize
        self.w_getter = w_getter

        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        # if input_shape[channel_axis] is None:
        #         raise ValueError('The channel dimension of the inputs '
        #                          'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
            
        base = self.kernel_size[0] * self.kernel_size[1]
        # if self.H == 'Glorot':
        #     nb_input = int(input_dim * base)
        #     nb_output = int(self.filters * base)
        #     self.H = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
        #     #print('Glorot H: {}'.format(self.H))
            
        # if self.kernel_lr_multiplier == 'Glorot':
        #     nb_input = int(input_dim * base)
        #     nb_output = int(self.filters * base)
        #     self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5/ (nb_input + nb_output)))
        #     #print('Glorot learning rate multiplier: {}'.format(self.lr_multiplier))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)

        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint, set_weight = self.w_getter)
        if self.use_bias:
            # self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight((self.output_dim,),
                                     initializer=self.bias_initializers,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        else:
            # self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        ternary_kernel = self.kernel
        

        if(self.ternarize):
            # print("------Ternarize Weights------")
            ternary_kernel = ternarize(ternary_kernel, H=self.H)
        else:
            # print("------ No Ternarize Weights------")
            ternary_kernel = K.cast(self.kernel, 'float16')
            ternary_kernel = K.cast(ternary_kernel, 'float32')

        outputs = K.conv2d(
            inputs,
            ternary_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
        
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(TernaryConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DepthwiseTernaryConv2D(SeparableConv2D):
    '''Quantized Convolution2D layer
    References: 
    "quantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, filters,strides, kernel_lr_multiplier='Glorot', 
                 bias_lr_multiplier=None, H=1., w_getter = None, **kwargs):
        super(DepthwiseTernaryConv2D, self).__init__(filters, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        self.strides = strides
        self.depthwise_mul = kwargs.get('depth_multiplier',1)
        self.w_getter = w_getter
        
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        depthwise_kernel_shape = self.kernel_size + (input_dim, self.depthwise_mul)
        pointwise_kernel_shape = [1,1,input_dim*self.depthwise_mul, self.filters]

        base = self.kernel_size[0] * self.kernel_size[1]
        if self.H == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.H = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
            #print('Glorot H: {}'.format(self.H))
            
        if self.kernel_lr_multiplier == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5/ (nb_input + nb_output)))
            #print('Glorot learning rate multiplier: {}'.format(self.lr_multiplier))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        # self.kernel_regularizer = regularizers.l2(0.000001)

        kwargs = {'set_weight' : self.w_getter[0]}
        self.depthwise_kernel = self.add_weight(shape=depthwise_kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint, **kwargs)

        kwargs = {'set_weight' : self.w_getter[1]}
        self.pointwise_kernel = self.add_weight(shape=pointwise_kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint, **kwargs)

        if self.use_bias:
            # self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight((self.output_dim,),
                                     initializer=self.bias_initializers,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        else:
            # self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # var = tf.Variable([-1.25, -1 , -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25])
        # self.depthwise_kernel = tf.Print(self.depthwise_kernel, [var], "Debugging ternarize", first_n = 2, summarize =20)

        # self.depthwise_kernel = tf.Print(self.depthwise_kernel, [self.depthwise_kernel], "----- Conv2d-dep kernel-----", first_n = 3, summarize = 200)
        # self.pointwise_kernel = tf.Print(self.pointwise_kernel, [self.pointwise_kernel], "----- Conv2d-point kernel-----", first_n = 3, summarize = 200)
        
        depthwise_ternarized_kernel = ternarize(self.depthwise_kernel, H=self.H)
        pointwise_ternarized_kernel = ternarize(self.pointwise_kernel, H=self.H) 
        # var = ternarize(var, H=self.H)
        # depthwise_ternarized_kernel = tf.Print(self.depthwise_kernel, [var], "Debugging ternarized", first_n = 2, summarize =20)

        # depthwise_ternarized_kernel = tf.Print(depthwise_ternarized_kernel, [depthwise_ternarized_kernel], "----- Conv2d-dep ternarize kernel-----", first_n = 3, summarize = 200)
        # pointwise_ternarized_kernel = tf.Print(pointwise_ternarized_kernel, [pointwise_ternarized_kernel], "----- Conv2d-point ternarize kernel-----", first_n = 3, summarize = 200)

        outputs = K.separable_conv2d(
            inputs,
            depthwise_ternarized_kernel,
            pointwise_ternarized_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)


        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(QuantizedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

TernaryConvolution2D = TernaryConv2D