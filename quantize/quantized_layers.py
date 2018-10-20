from keras.layers import InputSpec, Dense, Conv2D, SimpleRNN, SeparableConv2D
import numpy as np
from keras import constraints
from keras import initializers
from quantize.quantized_ops import quantize
import keras.backend as K
import tensorflow as tf

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

class QuantizedDense(Dense):
    ''' Binarized Dense layer
    References: 
    "QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, units, H=1., kernel_lr_multiplier='Glorot', bias_lr_multiplier=None, w_getter = None, nb=16, **kwargs):
        super(QuantizedDense, self).__init__(units, **kwargs)
        self.H = H
        self.nb = nb
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        super(QuantizedDense, self).__init__(units, **kwargs)
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
            self.bias = self.add_weight(shape=(self.units,),
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
        quantized_kernel = quantize(self.kernel, nb=self.nb)
        output = K.dot(inputs, quantized_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)


        return output
        
        
    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class QuantizedConv2D(Conv2D):
    '''Binarized Convolution2D layer
    References: 
    "QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, filters, kernel_regularizer=None,activity_regularizer=None, kernel_lr_multiplier='Glorot',
                 bias_lr_multiplier=None, H=1., w_getter = None, nb=16,  **kwargs):
        super(QuantizedConv2D, self).__init__(filters, **kwargs)
        self.H = H
        self.nb = nb
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        self.activity_regularizer =activity_regularizer
        self.kernel_regularizer = kernel_regularizer
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
        #self.bias_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint, set_weight = self.w_getter)

        if self.use_bias:
            # self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight((self.filters,),
                                     initializer=self.bias_initializer,
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
        quantized_kernel = quantize(self.kernel, nb=self.nb)
        # This is used to scale the gradients during back-prop with the glorot scheme, removing it! 
        # inverse_kernel_lr_multiplier = 1./self.kernel_lr_multiplier
        # inputs_qnn_gradient = (inputs - (1. - 1./inverse_kernel_lr_multiplier) * K.stop_gradient(inputs))\
        #           * inverse_kernel_lr_multiplier

        outputs = K.conv2d(
            inputs,
            quantized_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        # outputs = (outputs_qnn_gradient - (1. - 1./self.kernel_lr_multiplier) * K.stop_gradient(outputs_qnn_gradient))\
        #           * self.kernel_lr_multiplier


        #outputs = outputs*K.mean(K.abs(self.kernel))

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

class DepthwiseQuantizedConv2D(SeparableConv2D):
    '''Quantized Convolution2D layer
    References: 
    "quantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, filters,strides, kernel_lr_multiplier='Glorot', 
                 bias_lr_multiplier=None, H=1., w_getter = None, nb = 16, **kwargs):
        super(DepthwiseQuantizedConv2D, self).__init__(filters, **kwargs)
        self.H = H
        self.nb = nb
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
        # self.depthwise_kernel = tf.Print(self.depthwise_kernel, [self.depthwise_kernel], "----- Conv2d-dep kernel-----", first_n = 3, summarize = 20)
        # self.pointwise_kernel = tf.Print(self.pointwise_kernel, [self.pointwise_kernel], "----- Conv2d-point kernel-----", first_n = 3, summarize = 20)
        
        depthwise_quantized_kernel = quantize(self.depthwise_kernel, nb=self.nb)
        pointwise_quantized_kernel = quantize(self.pointwise_kernel, nb=self.nb) 

        inverse_kernel_lr_multiplier = 1./self.kernel_lr_multiplier
        inputs_qnn_gradient = (inputs - (1. - 1./inverse_kernel_lr_multiplier) * K.stop_gradient(inputs))\
                  * inverse_kernel_lr_multiplier

        outputs_qnn_gradient = K.separable_conv2d(
            inputs_qnn_gradient,
            depthwise_quantized_kernel,
            pointwise_quantized_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        outputs = (outputs_qnn_gradient - (1. - 1./self.kernel_lr_multiplier) * K.stop_gradient(outputs_qnn_gradient))\
                  * self.kernel_lr_multiplier

        # depthwise_quantized_kernel = tf.Print(depthwise_quantized_kernel, [depthwise_quantized_kernel], "----- Conv2d-dep quantize kernel-----", first_n = 3, summarize = 20)
        # pointwise_quantized_kernel = tf.Print(pointwise_quantized_kernel, [pointwise_quantized_kernel], "----- Conv2d-point quantize kernel-----", first_n = 3, summarize = 20)

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
        base_config = super(DepthwiseQuantizedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


