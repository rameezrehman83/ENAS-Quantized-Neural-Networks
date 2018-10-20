import os
import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
from models import Model
from common_ops import batch_norm
from common_ops import drop_path
from common_ops import global_avg_pool
from binarize.binary_ops import binary_tanh as binary_tanh_op
from binarize.binary_layers import BinaryDense, BinaryConv2D, DepthwiseBinaryConv2D
from ternarize.ternary_layers import TernaryDense, TernaryConv2D, DepthwiseTernaryConv2D
from quantize.quantized_layers import QuantizedConv2D, QuantizedDense, DepthwiseQuantizedConv2D
from keras.layers import Activation
from utils import count_model_params
from utils import get_train_ops
from common_ops import create_weight
from keras.activations import relu
from quantize.quantized_ops import quantized_relu as quantized_relu_op
from quantize.quantized_ops import quantized_tanh as quantized_tanh_op
from ternarize.ternary_ops import ternarize as ternarize_op
from functools import partial


# def ternary_tanh(x):
#     x = K.clip(x, -1, 1)
#     return ternarize_op(x)


H = 1.
kernel_lr_multiplier = 'Glorot'
use_bias = False
binarize = True
n_bits = 4

print("-"*80)
print("Nbits: ", n_bits)
print("-"*80)

# def bin_conv2d(x, w_getter, conv_kernel_size, conv_num_filters, conv_strides=1, use_activation = False, padding = 'same'):
#     x = tf.Print(x, [x], "-----Conv2d Input-----", first_n = 3, summarize = 20)
#     x = BinaryConv2D(conv_num_filters, kernel_size=(conv_kernel_size,conv_kernel_size), 
#                            data_format='channels_last', strides=(conv_strides,conv_strides),
#                            H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
#                            padding=padding, use_bias=use_bias, binarize = binarize, w_getter = w_getter)(x)
#     x = tf.Print(x, [x], "-----Conv2d output-----", first_n = 3, summarize = 20)

#     if(use_activation):
#         return Activation(binary_tanh_op)(x)
#     else: 
#         return x

# def bin_activation(x):
#     return Activation(binary_tanh_op)(x)

# def bin_dep_conv2d(x, w_getter, conv_kernel_size, conv_num_filters, conv_strides=1, use_activation = False, padding = 'same'):
#     x = tf.Print(x, [x], "-----Dep-Conv2d Input-----", first_n = 3, summarize = 20)
#     x = DepthwiseBinaryConv2D(conv_num_filters, kernel_size=(conv_kernel_size,conv_kernel_size), 
#                            data_format='channels_last', strides=(conv_strides,conv_strides),
#                            H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
#                            padding=padding, use_bias=use_bias, w_getter = w_getter)(x)
#     x = tf.Print(x, [x], "-----Dep-Conv2d output-----", first_n = 3, summarize = 20)

#     if(use_activation):
#         return Activation(binary_tanh_op)(x)
#     else: 
#         return x

# def bin_dense(x, units, w_getter, use_activation = False):
#     x = tf.Print(x, [x], "-----Dense Input-----", first_n = 3, summarize = 20)
#     x = BinaryDense(units, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, w_getter = w_getter)(x)
#     x = tf.Print(x, [x], "-----Dense output-----", first_n = 3, summarize = 20)
    
#     if(use_activation):
#         return Activation(binary_tanh_op)(x)
#     else: 
#         return x

def ter_conv2d(x, w_getter, conv_kernel_size, conv_num_filters, conv_strides=1, use_activation = False, padding = 'same'):
    # x = tf.Print(x, [x], "-----Conv2d Input-----", first_n = 3, summarize = 20)
    x = TernaryConv2D(conv_num_filters, kernel_size=(conv_kernel_size,conv_kernel_size), 
                           data_format='channels_last', strides=(conv_strides,conv_strides),
                           H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                           padding=padding, use_bias=use_bias, w_getter = w_getter)(x)
    # x = tf.Print(x, [x], "-----Conv2d output-----", first_n = 3, summarize = 20)

    if(use_activation):
        return Activation(binary_tanh_op)(x)
    else: 
        return x

def ter_activation(x):
    return Activation(binary_tanh_op)(x)

def ter_dep_conv2d(x, w_getter, conv_kernel_size, conv_num_filters, conv_strides=1, use_activation = False, padding = 'same'):
    # x = tf.Print(x, [x], "-----Dep-Conv2d Input-----", first_n = 3, summarize = 20)
    x = DepthwiseTernaryConv2D(conv_num_filters, kernel_size=(conv_kernel_size,conv_kernel_size), 
                           data_format='channels_last', strides=(conv_strides,conv_strides),
                           H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                           padding=padding, use_bias=use_bias, w_getter = w_getter)(x)
    # x = tf.Print(x, [x], "-----Dep-Conv2d output-----", first_n = 3, summarize = 20)

    if(use_activation):
        return Activation(binary_tanh_op)(x)
    else: 
        return x

def ter_dense(x, units, w_getter, use_activation = False):
    # x = tf.Print(x, [x], "-----Dense Input-----", first_n = 3, summarize = 20)
    x = TernaryDense(units, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, w_getter = w_getter)(x)
    # x = tf.Print(x, [x], "-----Dense output-----", first_n = 3, summarize = 20)
    
    if(use_activation):
        return Activation(binary_tanh_op)(x)
    else: 
        return x

def quant_conv2d(x, w_getter, conv_kernel_size, conv_num_filters, nb = n_bits, conv_strides=1, use_activation = False, padding = 'same'):
    # x = tf.Print(x, [x], "-----Conv2d Input-----", first_n = 3, summarize = 20)
    x = QuantizedConv2D(conv_num_filters, kernel_size=(conv_kernel_size,conv_kernel_size), 
                           data_format='channels_last', strides=(conv_strides,conv_strides),
                           H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                           padding=padding, use_bias=use_bias, w_getter = w_getter, nb = nb)(x)
    # x = tf.Print(x, [x], "-----Conv2d output-----", first_n = 3, summarize = 20)

    if(use_activation):
        return Activation(partial(quantized_tanh_op, nb = nb))(x)
    else: 
        return x

def quant_activation(x):
    return Activation(partial(quantized_tanh_op, nb = n_bits))(x)

def quant_dep_conv2d(x, w_getter, conv_kernel_size, conv_num_filters, nb = n_bits, conv_strides=1, use_activation = False, padding = 'same'):
    # x = tf.Print(x, [x], "-----Dep-Conv2d Input-----", first_n = 3, summarize = 20)
    x = DepthwiseQuantizedConv2D(conv_num_filters, kernel_size=(conv_kernel_size,conv_kernel_size), 
                           data_format='channels_last', strides=(conv_strides,conv_strides),
                           H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                           padding=padding, use_bias=use_bias, w_getter = w_getter, nb = nb)(x)
    # x = tf.Print(x, [x], "-----Dep-Conv2d output-----", first_n = 3, summarize = 20)

    if(use_activation):
        return Activation(partial(quantized_tanh_op, nb = nb))(x)
    else: 
        return x

def quant_dense(x, units, w_getter, nb = n_bits, use_activation = False):
    # x = tf.Print(x, [x], "-----Dense Input-----", first_n = 3, summarize = 20)
    x = QuantizedDense(units, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, w_getter = w_getter, nb = nb)(x)
    # x = tf.Print(x, [x], "-----Dense output-----", first_n = 3, summarize = 20)
    
    if(use_activation):
        return Activation(partial(quantized_tanh_op, nb = nb))(x)
    else: 
        return x




class MicroChild(Model):
    def __init__(self,
                 images,
                 labels,
                 use_aux_heads=False,
                 cutout_size=None,
                 fixed_arc=None,
                 num_layers=2,
                 num_cells=5,
                 out_filters=24,
                 keep_prob=1.0,
                 drop_path_keep_prob=None,
                 batch_size=32,
                 clip_mode=None,
                 grad_bound=None,
                 l2_reg=1e-4,
                 lr_init=0.1,
                 lr_dec_start=0,
                 lr_dec_every=10000,
                 lr_dec_rate=0.1,
                 lr_cosine=False,
                 lr_max=None,
                 lr_min=None,
                 lr_T_0=None,
                 lr_T_mul=None,
                 num_epochs=None,
                 optim_algo=None,
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 data_format="NHWC",
                 name="child",
                 channel = None,
                 **kwargs
                 ):

        super(self.__class__, self).__init__(
            images,
            labels,
            cutout_size=cutout_size,
            batch_size=batch_size,
            clip_mode=clip_mode,
            grad_bound=grad_bound,
            l2_reg=l2_reg,
            lr_init=lr_init,
            lr_dec_start=lr_dec_start,
            lr_dec_every=lr_dec_every,
            lr_dec_rate=lr_dec_rate,
            keep_prob=keep_prob,
            optim_algo=optim_algo,
            sync_replicas=sync_replicas,
            num_aggregate=num_aggregate,
            num_replicas=num_replicas,
            data_format=data_format,
            name=name)

        if self.data_format == "NHWC":
            self.actual_data_format = "channels_last"
        elif self.data_format == "NCHW":
            self.actual_data_format = "channels_first"
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        self.use_aux_heads = use_aux_heads
        self.num_epochs = num_epochs
        self.num_train_steps = self.num_epochs * self.num_train_batches
        self.drop_path_keep_prob = drop_path_keep_prob
        self.lr_cosine = lr_cosine
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.out_filters = out_filters
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.fixed_arc = fixed_arc
        self.channel = channel

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step")

        if self.drop_path_keep_prob is not None:
            assert num_epochs is not None, "Need num_epochs to drop_path"
        
        pool_distance = self.num_layers //4
        self.pool_layers = [pool_distance, 2 * pool_distance + 1, self.num_layers]
        
        '''
        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2 * pool_distance + 1]
        '''

        if self.use_aux_heads:
            self.aux_head_indices = [self.pool_layers[-2] + 1]

    def _factorized_reduction(self, x, out_filters, stride,
                              is_training, type_op = "ternarize"): 
        """Reduces the shape of x without information loss due to striding."""
        assert out_filters % 2 == 0, (
            "Need even number of filters when using this factorized reduction.")
        if stride == 1:
            with tf.variable_scope("path_conv"):
                inp_c = self._get_C(x)
                w = create_weight("w", [1, 1, inp_c, out_filters])
                if(type_op == "quantize"):
                    x = quant_conv2d(x, w, 1, out_filters)
                    # x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                    #                  data_format=self.data_format)
                else:
                    x = ter_conv2d(x, w, 1, out_filters)

                x = batch_norm(x, is_training, data_format=self.data_format)
                return x

        stride_spec = self._get_strides(stride)  # [1,2,2,1]
        # Skip path 1
        x = tf.identity(x)
        path1 = tf.nn.avg_pool(
            x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)

        with tf.variable_scope("path1_conv"):
            inp_c = self._get_C(path1)  
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            if(type_op == "quantize"):
                path1 = quant_conv2d(path1, w, 1, out_filters//2, padding = 'valid')
            else:
                path1 = ter_conv2d(path1, w, 1, out_filters//2, padding = 'valid')

            # path1 = tf.nn.conv2d(path1, w, [1, 1, 1, 1], "VALID",
            #                      data_format=self.data_format)

        # Skip path 2
        # First pad with 0"s on the right and bottom, then shift the filter to
        # include those 0"s that were added.
        if self.data_format == "NHWC":
            pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
            path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
            concat_axis = 3
        else:
            pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
            path2 = tf.pad(x, pad_arr)[:, :, 1:, 1:]
            concat_axis = 1

        path2 = tf.nn.avg_pool(
            path2, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
        with tf.variable_scope("path2_conv"):
            inp_c = self._get_C(path2)
            w = create_weight("w", [1, 1, inp_c, out_filters // 2])
            if(type_op == "quantize"):
                path2 = quant_conv2d(path2, w, 1, out_filters//2, padding = 'valid')
            else:
                path2 = ter_conv2d(path2, w, 1, out_filters//2, padding = 'valid')
            # path2 = bin_conv2d(path2, w, 1, out_filters//2, padding = 'valid')

            # path2 = tf.nn.conv2d(path2, w, [1, 1, 1, 1], "VALID",
            #                      data_format=self.data_format)

        # Concat and apply BN
        final_path = tf.concat(values=[path1, path2], axis=concat_axis)
        final_path = batch_norm(final_path, is_training,
                                data_format=self.data_format)

        return final_path

    def _get_C(self, x):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            return x.get_shape()[3].value
        elif self.data_format == "NCHW":
            return x.get_shape()[1].value
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _get_HW(self, x):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        return x.get_shape()[2].value

    def _get_strides(self, stride):
        """
        Args:
          x: tensor of shape [N, H, W, C] or [N, C, H, W]
        """
        if self.data_format == "NHWC":
            return [1, stride, stride, 1]
        elif self.data_format == "NCHW":
            return [1, 1, stride, stride]
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _apply_drop_path(self, x, layer_id):
        drop_path_keep_prob = self.drop_path_keep_prob

        layer_ratio = float(layer_id + 1) / (self.num_layers + 2)
        drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)

        step_ratio = tf.to_float(self.global_step + 1) / tf.to_float(self.num_train_steps)
        step_ratio = tf.minimum(1.0, step_ratio)
        drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)

        x = drop_path(x, drop_path_keep_prob)
        return x

    def _maybe_calibrate_size(self, layers, out_filters, is_training, type_op = "ternarize"): 
        """Makes sure layers[0] and layers[1] have the same shapes."""
        hw = [self._get_HW(layer) for layer in layers]  
        c = [self._get_C(layer) for layer in layers]  
        print("hw", hw)
        print("c", c)

        with tf.variable_scope("calibrate"):
            x = layers[0]  
            if hw[0] != hw[1]:  
                # assert hw[0] == 2 * hw[1]  
                with tf.variable_scope("pool_x"):
                    # x = tf.nn.relu(x)
                    if(type_op == "quantize"):
                        x = quant_activation(x)
                        x = self._factorized_reduction(x, out_filters, 2, is_training, type_op = type_op)
                    else:
                        x = ter_activation(x)
                        x = self._factorized_reduction(x, out_filters, 2, is_training, type_op = type_op)


            elif c[0] != out_filters:  
                with tf.variable_scope("pool_x"):
                    w = create_weight("w", [1, 1, c[0], out_filters])
                    # x = tf.nn.relu(x)
                    if(type_op == "quantize"):
                        x = quant_activation(x)
                        x = quant_conv2d(x, w, 1, out_filters)
                    else:
                        x = ter_activation(x)
                        x = ter_conv2d(x, w, 1, out_filters)

                    # x = bin_activation(x)
                    # x = bin_conv2d(x, w, 1, out_filters)
                    # x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                    #                  data_format=self.data_format)
                    x = batch_norm(x, is_training, data_format=self.data_format)  

            y = layers[1]  
            if c[1] != out_filters:  
                with tf.variable_scope("pool_y"):
                    w = create_weight("w", [1, 1, c[1], out_filters])
                    # y = tf.nn.relu(y)
                    if(type_op == "quantize"):
                        y = quant_activation(y)
                        y = quant_conv2d(y, w, 1, out_filters)
                    else:
                        y = ter_activation(y)
                        y = ter_conv2d(y, w, 1, out_filters)

                    # y = bin_activation(y)
                    # y = bin_conv2d(y, w, 1, out_filters)

                    # y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                    #                  data_format=self.data_format)
                    y = batch_norm(y, is_training, data_format=self.data_format)
        return [x, y]

    def _model(self, images, is_training, reuse=False):
        """Compute the logits given the images."""
        if self.fixed_arc is None:
            is_training = True

        with tf.variable_scope(self.name, reuse=reuse):
            with tf.variable_scope("stem_conv"):
                w = create_weight("w", [3, 3, self.channel, self.out_filters * 3])  
                x = quant_conv2d(images, w, 3, self.out_filters*3)
                # x = tf.nn.conv2d(
                #     images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)
            if self.data_format == "NHWC":
                split_axis = 3
            elif self.data_format == "NCHW":
                split_axis = 1
            else:
                raise ValueError("Unknown data_format '{0}'".format(self.data_format))
            layers = [x, x]

            # building layers in the micro space
            out_filters = self.out_filters 
            for layer_id in range(self.num_layers + 2):  
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    if(layer_id == 0):
                        with tf.variable_scope("quantize"):
                            if layer_id not in self.pool_layers: 
                                if self.fixed_arc is None:
                                    x = self._enas_layer(
                                        layer_id, layers, self.normal_arc, out_filters, type_op = "quantize")  

                                else:
                                    x = self._fixed_layer(
                                        layer_id, layers, self.normal_arc, out_filters, 1, is_training,
                                        normal_or_reduction_cell="normal", type_op = "quantize")
                            else:  
                                out_filters *= 2  
                                if self.fixed_arc is None:
                                    x = self._factorized_reduction(x, out_filters, 2, is_training, type_op = "quantize")
                                    layers = [layers[-1], x]  
                                    x = self._enas_layer(
                                        layer_id, layers, self.reduce_arc, out_filters, type_op = "quantize")
                                else:
                                    x = self._fixed_layer(
                                        layer_id, layers, self.reduce_arc, out_filters, 2, is_training,
                                        normal_or_reduction_cell="reduction", type_op = "quantize")

                            print("Layer {0:>2d}: {1}".format(layer_id, x))
                            layers = [layers[-1], x]
                    else:
                        with tf.variable_scope("ternarize"): 
                            if layer_id not in self.pool_layers: 
                                if self.fixed_arc is None:
                                    x = self._enas_layer(
                                        layer_id, layers, self.normal_arc, out_filters)  

                                else:
                                    x = self._fixed_layer(
                                        layer_id, layers, self.normal_arc, out_filters, 1, is_training,
                                        normal_or_reduction_cell="normal")
                            else:  
                                out_filters *= 2  
                                if self.fixed_arc is None:
                                    x = self._factorized_reduction(x, out_filters, 2, is_training)
                                    layers = [layers[-1], x]  
                                    x = self._enas_layer(
                                        layer_id, layers, self.reduce_arc, out_filters)
                                else:
                                    x = self._fixed_layer(
                                        layer_id, layers, self.reduce_arc, out_filters, 2, is_training,
                                        normal_or_reduction_cell="reduction")

                            print("Layer {0:>2d}: {1}".format(layer_id, x))
                            layers = [layers[-1], x]

                # auxiliary heads
                self.num_aux_vars = 0
                if (self.use_aux_heads and
                        layer_id in self.aux_head_indices
                        and is_training):
                    print("Using aux_head at layer {0}".format(layer_id))
                    with tf.variable_scope("aux_head"):
                        # aux_logits = tf.nn.relu(x)
                        aux_logits = tf.identity(ter_activation(x))

                        if (aux_logits.get_shape()[2].value - 3) % 5 ==0:
                            aux_logits = tf.layers.average_pooling2d(
                                aux_logits, [5, 5], [3, 3], "VALID",
                                data_format=self.actual_data_format)
                        else:
                            aux_logits = tf.layers.average_pooling2d(
                                aux_logits, [5,5], [3,3], "SAME",
                                data_format = self.actual_data_format)

                        with tf.variable_scope("proj"):
                            inp_c = self._get_C(aux_logits)
                            w = create_weight("w", [1, 1, inp_c, 128])
                            aux_logits = ter_conv2d(aux_logits, w, 1, 128)
                            # aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                            #                           data_format=self.data_format)
                            aux_logits = batch_norm(aux_logits, is_training=True,
                                                    data_format=self.data_format)
                            # aux_logits = tf.nn.relu(aux_logits)
                            aux_logits = ter_activation(aux_logits)

                        with tf.variable_scope("avg_pool"):
                            inp_c = self._get_C(aux_logits)
                            hw = self._get_HW(aux_logits)
                            w = create_weight("w", [hw, hw, inp_c, 768])
                            aux_logits = ter_conv2d(aux_logits, w, hw, 768)

                            # aux_logits = tf.nn.conv2d(aux_logits, w, [1, 1, 1, 1], "SAME",
                            #                           data_format=self.data_format)
                            aux_logits = batch_norm(aux_logits, is_training=True,
                                                    data_format=self.data_format)
                            # aux_logits = tf.nn.relu(aux_logits)
                            aux_logits = ter_activation(aux_logits)

                        with tf.variable_scope("fc"):
                            aux_logits = global_avg_pool(aux_logits,
                                                         data_format=self.data_format)
                            inp_c = aux_logits.get_shape()[1].value
                            w = create_weight("w", [inp_c, 10])
                            aux_logits = tf.matmul(aux_logits, w)
                            self.aux_logits = aux_logits

                    aux_head_variables = [
                        var for var in tf.trainable_variables() if (
                                var.name.startswith(self.name) and "aux_head" in var.name)]
                    self.num_aux_vars = count_model_params(aux_head_variables)
                    print("Aux head uses {0} params".format(self.num_aux_vars))

            # x = tf.nn.relu(x)
            x = ter_activation(x)

            x = global_avg_pool(x, data_format=self.data_format)
            if is_training and self.keep_prob is not None and self.keep_prob < 1.0:
                x = tf.nn.dropout(x, self.keep_prob)
            with tf.variable_scope("fc"):
                inp_c = x.get_shape()[1].value
                w = create_weight("w", [inp_c, 10])
                x = tf.matmul(x, w)
        return x

    def _fixed_conv(self, x, f_size, out_filters, stride, is_training,
                    stack_convs=2, type_op = "ternarize"):
        """Apply fixed convolution.

        Args:
          stacked_convs: number of separable convs to apply.
        """

        for conv_id in range(stack_convs):
            inp_c = self._get_C(x)
            if conv_id == 0:
                strides = self._get_strides(stride)
            else:
                strides = [1, 1, 1, 1]
                stride = 1

            with tf.variable_scope("sep_conv_{}".format(conv_id)):
                w_depthwise = create_weight("w_depth", [f_size, f_size, inp_c, 1])
                w_pointwise = create_weight("w_point", [1, 1, inp_c, out_filters])
                # x = tf.nn.relu(x)
                print("f_size: ", f_size)
                if(type_op == "quantize"):
                    x = quant_activation(x)
                    x = quant_dep_conv2d(x, [w_depthwise, w_pointwise], f_size, out_filters, conv_strides = stride)
                else:
                    x = ter_activation(x)
                    x = ter_dep_conv2d(x, [w_depthwise, w_pointwise], f_size, out_filters, conv_strides = stride)

                # x = tf.nn.separable_conv2d(
                #     x,
                #     depthwise_filter=w_depthwise,
                #     pointwise_filter=w_pointwise,
                #     strides=strides, padding="SAME", data_format=self.data_format)
                x = batch_norm(x, is_training, data_format=self.data_format)

        return x

    def _fixed_combine(self, layers, used, out_filters, is_training,
                       normal_or_reduction_cell="normal", type_op = "ternarize"):
        """Adjust if necessary.

        Args:
          layers: a list of tf tensors of size [NHWC] of [NCHW].
          used: a numpy tensor, [0] means not used.
        """

        out_hw = min([self._get_HW(layer)
                      for i, layer in enumerate(layers) if used[i] == 0])
        out = []

        with tf.variable_scope("final_combine"):
            for i, layer in enumerate(layers):
                if used[i] == 0:
                    hw = self._get_HW(layer)
                    if hw > out_hw:
                        assert hw == out_hw * 2, ("i_hw={0} != {1}=o_hw".format(hw, out_hw))
                        with tf.variable_scope("calibrate_{0}".format(i)):
                            x = self._factorized_reduction(layer, out_filters, 2, is_training, type_op = type_op)
                    else:
                        x = layer
                    out.append(x)

            if self.data_format == "NHWC":
                out = tf.concat(out, axis=3)
            elif self.data_format == "NCHW":
                out = tf.concat(out, axis=1)
            else:
                raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        return out

    def _fixed_layer(self, layer_id, prev_layers, arc, out_filters, stride,
                     is_training, normal_or_reduction_cell="normal", type_op = "ternarize"):
        """
        Args:
          prev_layers: cache of previous layers. for skip connections
          is_training: for batch_norm
        """

        assert len(prev_layers) == 2
        layers = [prev_layers[0], prev_layers[1]]
        layers = self._maybe_calibrate_size(layers, out_filters,
                                            is_training=is_training, type_op = type_op)

        with tf.variable_scope("layer_base"):
            x = layers[1]
            inp_c = self._get_C(x)
            w = create_weight("w", [1, 1, inp_c, out_filters])
            # x = tf.nn.relu(x)
            if(type_op == "quantize"):
                x = quant_activation(x)
                x = quant_conv2d(x,w, 1, out_filters)
            else:
                x = ter_activation(x)
                x = ter_conv2d(x,w, 1, out_filters)
            # x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
            #                  data_format=self.data_format)
            x = batch_norm(x, is_training, data_format=self.data_format)
            layers[1] = x

        used = np.zeros([self.num_cells + 2], dtype=np.int32)
        f_sizes = [3, 5]
        for cell_id in range(self.num_cells):
            with tf.variable_scope("cell_{}".format(cell_id)):
                x_id = arc[4 * cell_id]
                used[x_id] += 1
                x_op = arc[4 * cell_id + 1]
                x = layers[x_id]
                x_stride = stride if x_id in [0, 1] else 1
                with tf.variable_scope("x_conv"):
                    if x_op in [0, 1]:
                        f_size = f_sizes[x_op]
                        x = self._fixed_conv(x, f_size, out_filters, x_stride, is_training, type_op = type_op)
                    elif x_op in [2, 3]:
                        inp_c = self._get_C(x)
                        if x_op == 2:
                            x = tf.identity(x)
                            print("type x:", x)
                            x = tf.layers.average_pooling2d(
                                x, [3, 3], [x_stride, x_stride], "SAME",
                                data_format=self.actual_data_format)
                        else:
                            x = tf.identity(x)
                            print("type x:", x)
                            x = tf.layers.max_pooling2d(
                                x, [3, 3], [x_stride, x_stride], "SAME",
                                data_format=self.actual_data_format)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            # x = tf.nn.relu(x)
                            if(type_op == "quantize"):
                                x = quant_activation(x)
                                x = quant_conv2d(x,w, 1, out_filters)
                            else:
                                x = ter_activation(x)
                                x = ter_conv2d(x,w, 1, out_filters)
                            # x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                            #                  data_format=self.data_format)
                            x = batch_norm(x, is_training, data_format=self.data_format)
                    else:
                        inp_c = self._get_C(x)
                        if x_stride > 1:
                            assert x_stride == 2
                            x = self._factorized_reduction(x, out_filters, 2, is_training, type_op = type_op)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            # x = tf.nn.relu(x)
                            if(type_op == "quantize"):
                                x = quant_activation(x)
                                x = quant_conv2d(x,w, 1, out_filters)
                            else:
                                x = ter_activation(x)
                                x = ter_conv2d(x,w, 1, out_filters)
                            # x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                            #  data_format=self.data_format)
                            x = batch_norm(x, is_training, data_format=self.data_format)
                    if (x_op in [0, 1, 2, 3] and
                            self.drop_path_keep_prob is not None and
                            is_training):
                        x = self._apply_drop_path(x, layer_id)

                y_id = arc[4 * cell_id + 2]
                used[y_id] += 1
                y_op = arc[4 * cell_id + 3]
                y = layers[y_id]
                y_stride = stride if y_id in [0, 1] else 1
                with tf.variable_scope("y_conv"):
                    if y_op in [0, 1]:
                        f_size = f_sizes[y_op]
                        y = self._fixed_conv(y, f_size, out_filters, y_stride, is_training)
                    elif y_op in [2, 3]:
                        inp_c = self._get_C(y)
                        if y_op == 2:
                            y = tf.identity(y)
                            y = tf.layers.average_pooling2d(
                                y, [3, 3], [y_stride, y_stride], "SAME",
                                data_format=self.actual_data_format)
                        else:
                            y = tf.identity(y)
                            y = tf.layers.max_pooling2d(
                                y, [3, 3], [y_stride, y_stride], "SAME",
                                data_format=self.actual_data_format)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            # y = tf.nn.relu(y)
                            if(type_op == "quantize"):
                                x = quant_activation(x)
                                x = quant_conv2d(x,w, 1, out_filters)
                            else:
                                x = ter_activation(x)
                                x = ter_conv2d(x,w, 1, out_filters)
                            # y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                            #                  data_format=self.data_format)
                            y = batch_norm(y, is_training, data_format=self.data_format)
                    else:
                        inp_c = self._get_C(y)
                        if y_stride > 1:
                            assert y_stride == 2
                            y = self._factorized_reduction(y, out_filters, 2, is_training)
                        if inp_c != out_filters:
                            w = create_weight("w", [1, 1, inp_c, out_filters])
                            # y = tf.nn.relu(y)
                            if(type_op == "quantize"):
                                x = quant_activation(x)
                                x = quant_conv2d(x,w, 1, out_filters)
                            else:
                                x = ter_activation(x)
                                x = ter_conv2d(x,w, 1, out_filters)
                            # y = tf.nn.conv2d(y, w, [1, 1, 1, 1], "SAME",
                            #                  data_format=self.data_format)
                            y = batch_norm(y, is_training, data_format=self.data_format)

                    if (y_op in [0, 1, 2, 3] and
                            self.drop_path_keep_prob is not None and
                            is_training):
                        y = self._apply_drop_path(y, layer_id)

                out = x + y
                layers.append(out)
        out = self._fixed_combine(layers, used, out_filters, is_training,
                                  normal_or_reduction_cell, type_op = type_op)

        return out

    def _enas_cell(self, x, curr_cell, prev_cell, op_id, out_filters, type_op = "ternarize"):
        """Performs an enas operation specified by op_id."""

        num_possible_inputs = curr_cell + 1
        x = tf.identity(x)

        with tf.variable_scope("avg_pool"):
            avg_pool = tf.layers.average_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
            avg_pool_c = self._get_C(avg_pool)
            if avg_pool_c != out_filters:
                with tf.variable_scope("conv"):
                    w = create_weight(
                        "w", [num_possible_inputs, avg_pool_c * out_filters])
                    w = w[prev_cell]
                    w = tf.reshape(w, [1, 1, avg_pool_c, out_filters])
                    # avg_pool = tf.nn.relu(avg_pool)

                    if(type_op == "quantize"):
                        avg_pool = quant_activation(avg_pool)
                        avg_pool = quant_conv2d(avg_pool, w, 1, out_filters)
                    else:
                        avg_pool = ter_activation(avg_pool)
                        avg_pool = ter_conv2d(avg_pool, w, 1, out_filters)
                    # avg_pool = tf.nn.conv2d(avg_pool, w, strides=[1, 1, 1, 1],
                    #                         padding="SAME", data_format=self.data_format)
                    avg_pool = batch_norm(avg_pool, is_training=True,
                                          data_format=self.data_format)

        with tf.variable_scope("max_pool"):
            max_pool = tf.layers.max_pooling2d(
                x, [3, 3], [1, 1], "SAME", data_format=self.actual_data_format)
            max_pool_c = self._get_C(max_pool)
            if max_pool_c != out_filters:
                with tf.variable_scope("conv"):
                    w = create_weight(
                        "w", [num_possible_inputs, max_pool_c * out_filters])
                    w = w[prev_cell]
                    w = tf.reshape(w, [1, 1, max_pool_c, out_filters])
                    # max_pool = tf.nn.relu(max_pool)
                    if(type_op == "quantize"):
                        max_pool = quant_activation(max_pool)
                        max_pool = quant_conv2d(max_pool, w, 1, out_filters)
                    else:
                        max_pool = ter_activation(max_pool)
                        max_pool = ter_conv2d(max_pool, w, 1, out_filters)
                    # max_pool = tf.nn.conv2d(max_pool, w, strides=[1, 1, 1, 1],
                    #                         padding="SAME", data_format=self.data_format)
                    max_pool = batch_norm(max_pool, is_training=True,
                                          data_format=self.data_format)

        x_c = self._get_C(x)
        if x_c != out_filters:
            with tf.variable_scope("x_conv"):
                w = create_weight("w", [num_possible_inputs, x_c * out_filters])
                w = w[prev_cell]
                w = tf.reshape(w, [1, 1, x_c, out_filters])
                # x = tf.nn.relu(x)
                if(type_op == "quantize"):
                    x = quant_activation(x)
                    x = quant_conv2d(x, w, 1, out_filters)
                else:
                    x = ter_activation(x)
                    x = ter_conv2d(x, w, 1, out_filters)
                # x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME",
                #                  data_format=self.data_format)
                x = batch_norm(x, is_training=True, data_format=self.data_format)

        out = [
            self._enas_conv(x, curr_cell, prev_cell, 3, out_filters, type_op = type_op),
            self._enas_conv(x, curr_cell, prev_cell, 5, out_filters, type_op = type_op),
            avg_pool,
            max_pool,
            x,
        ]

        out = tf.stack(out, axis=0)
        out = out[op_id, :, :, :, :]
        return out

    def _enas_conv(self, x, curr_cell, prev_cell, filter_size, out_filters, type_op = "ternarize",
                   stack_conv=2):
        """Performs an enas convolution specified by the relevant parameters."""

        with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
            num_possible_inputs = curr_cell + 2  
            for conv_id in range(stack_conv):
                with tf.variable_scope("stack_{0}".format(conv_id)):
                    # create params and pick the correct path
                    inp_c = self._get_C(x)  
                    w_depthwise = create_weight(
                        "w_depth", [num_possible_inputs, filter_size * filter_size * inp_c]) 
                    w_depthwise = w_depthwise[prev_cell, :]

                    w_depthwise = tf.reshape(
                        w_depthwise, [filter_size, filter_size, inp_c, 1])  

                    w_pointwise = create_weight(
                        "w_point", [num_possible_inputs, inp_c * out_filters])  
                    w_pointwise = w_pointwise[prev_cell, :]
                    w_pointwise = tf.reshape(w_pointwise, [1, 1, inp_c, out_filters])  

                    with tf.variable_scope("bn"):
                        zero_init = tf.initializers.zeros(dtype=tf.float32)
                        one_init = tf.initializers.ones(dtype=tf.float32)
                        offset = create_weight(
                            "offset", [num_possible_inputs, out_filters],  
                            initializer=zero_init)
                        scale = create_weight(
                            "scale", [num_possible_inputs, out_filters],  
                            initializer=one_init)
                        offset = offset[prev_cell]
                        scale = scale[prev_cell]

                    # the computations
                    # x = tf.nn.relu(x)
                    if(type_op == "quantize"):
                        x = quant_activation(x)
                        x = quant_dep_conv2d(x, [w_depthwise, w_pointwise], filter_size, out_filters)
                    else:
                        x = ter_activation(x)
                        x = ter_dep_conv2d(x, [w_depthwise, w_pointwise], filter_size, out_filters)

                    # x = tf.nn.separable_conv2d(
                    #     x,
                    #     depthwise_filter=w_depthwise,
                    #     pointwise_filter=w_pointwise,
                    #     strides=[1, 1, 1, 1], padding="SAME",
                    #     data_format=self.data_format)

                    x, _, _ = tf.nn.fused_batch_norm(
                        x, scale, offset, epsilon=1e-5, data_format=self.data_format,
                        is_training=True)
        return x

    def _enas_layer(self, layer_id, prev_layers, arc, out_filters, type_op = "ternarize"):
        """
        Args:
          layer_id: current layer
          prev_layers: cache of previous layers. for skip connections
          start_idx: where to start looking at. technically, we can infer this
            from layer_id, but why bother...
        """

        assert len(prev_layers) == 2, "need exactly 2 inputs"
        layers = [prev_layers[0], prev_layers[1]]
        layers = self._maybe_calibrate_size(layers, out_filters,
                                            is_training=True, type_op = type_op)  
        used = []
        for cell_id in range(self.num_cells):  
            prev_layers = tf.stack(layers, axis=0)  
            with tf.variable_scope("cell_{0}".format(cell_id)):
                with tf.variable_scope("x"):
                    x_id = arc[4 * cell_id]  
                    x_op = arc[4 * cell_id + 1] 
                    x = prev_layers[x_id, :, :, :, :]
                    x = self._enas_cell(x, cell_id, x_id, x_op, out_filters, type_op = type_op)
                    x_used = tf.one_hot(x_id, depth=self.num_cells + 2, dtype=tf.int32)

                with tf.variable_scope("y"):
                    y_id = arc[4 * cell_id + 2]
                    y_op = arc[4 * cell_id + 3]
                    y = prev_layers[y_id, :, :, :, :]
                    y = self._enas_cell(y, cell_id, y_id, y_op, out_filters, type_op = type_op)
                    y_used = tf.one_hot(y_id, depth=self.num_cells + 2,
                                        dtype=tf.int32)  

                out = x + y  
                used.extend([x_used, y_used])
                layers.append(out)  

        # Adds all tensors in used
        used = tf.add_n(used)  
        # ouput shape = (num_blocks+2,)
        indices = tf.where(tf.equal(used, 0))  
        # output shape = (?, 1)
        indices = tf.to_int32(indices)  
        # output shape = (?, 1) converted to int32
        indices = tf.reshape(indices, [-1])
        # output shape = (?, ) remove the uncessary axis
        
        num_outs = tf.size(indices)  
        out = tf.stack(layers, axis=0) 
        # arrange the prev two layers and block outputs in a single tensor  
        out = tf.gather(out, indices, axis=0)  
        # select the unused tensors from out, shape = (num_outs, shape(out) = [N, H, W, out_filters])

        inp = prev_layers[0]  
        if self.data_format == "NHWC":
            N = tf.shape(inp)[0]
            H = tf.shape(inp)[1]
            W = tf.shape(inp)[2]
            C = tf.shape(inp)[3]
            out = tf.transpose(out, [1, 2, 3, 0, 4])
            out = tf.reshape(out, [N, H, W, num_outs * out_filters])
        # reshape out to (N, H, W, out_filters*num_outs)
        elif self.data_format == "NCHW":
            N = tf.shape(inp)[0]
            C = tf.shape(inp)[1]
            H = tf.shape(inp)[2]
            W = tf.shape(inp)[3]
            out = tf.transpose(out, [1, 0, 2, 3, 4])  
            out = tf.reshape(out,
                             [N, num_outs * out_filters, H, W])  
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

        with tf.variable_scope("final_conv"):
            w = create_weight("w", [self.num_cells + 2, out_filters * out_filters])
            w = tf.gather(w, indices, axis=0)
            w = tf.reshape(w, [1, 1, num_outs * out_filters, out_filters])
            # out = tf.nn.relu(out)
            if(type_op == "quantize"):
                out = quant_activation(out)
                out = quant_conv2d(out, w, 1, out_filters)
            else:
                out = ter_activation(out)
                out = ter_conv2d(out, w, 1, out_filters)
            # out = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME",
            #                    data_format=self.data_format)
            out = batch_norm(out, is_training=True, data_format=self.data_format)

        out = tf.reshape(out, tf.shape(prev_layers[0]))

        return out

    # override
    def _build_train(self):
        print("-" * 80)
        print("Build train graph")
        logits = self._model(self.x_train, is_training=True)
        # logits = tf.reshape(logits, [-1, 10])
        self.y_train_hot = tf.one_hot(self.y_train, 10, dtype =tf.int32, on_value= 1, off_value= -1)
        print("Logits shape: ", logits)
        print("Y_train shape: ", self.y_train_hot)

        # using squared_hinge loss 
        self.loss = tf.reduce_mean(tf.losses.hinge_loss(self.y_train_hot, logits))

        # log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=logits, labels=self.y_train)  

        # self.loss = tf.reduce_mean(log_probs)  

        if self.use_aux_heads:
            # log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     logits=self.aux_logits, labels=self.y_train)
            # self.aux_loss = tf.reduce_mean(log_probs)
            self.aux_loss = tf.losses.hinge_loss(self.y_train_hot, self.aux_logits, reduction = tf.losses.Reduction.MEAN)
            train_loss = self.loss + 0.4 * self.aux_loss
        else:
            train_loss = self.loss

        self.train_preds = tf.argmax(logits, axis=1)
        self.train_preds = tf.to_int32(self.train_preds)

        self.train_acc = tf.equal(self.train_preds, self.y_train)
        self.train_acc = tf.to_int32(self.train_acc)
        self.train_acc = tf.reduce_sum(self.train_acc)  

        tf_variables = [
            var for var in tf.trainable_variables() if (
                    var.name.startswith(self.name) and "aux_head" not in var.name)]
        self.num_vars = count_model_params(tf_variables)
        print("Model has {0} params".format(self.num_vars))

        self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
            train_loss,
            tf_variables,
            self.global_step,
            clip_mode=self.clip_mode,
            grad_bound=self.grad_bound,
            l2_reg=self.l2_reg,
            lr_init=self.lr_init,
            lr_dec_start=self.lr_dec_start,
            lr_dec_every=self.lr_dec_every,
            lr_dec_rate=self.lr_dec_rate,
            lr_cosine=self.lr_cosine,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            lr_T_0=self.lr_T_0,
            lr_T_mul=self.lr_T_mul,
            num_train_batches=self.num_train_batches,
            optim_algo=self.optim_algo,
            sync_replicas=self.sync_replicas,
            num_aggregate=self.num_aggregate,
            num_replicas=self.num_replicas)


    # override
    def _build_valid(self):
        if self.x_valid is not None:
            print("-" * 80)
            print("Build valid graph")
            logits = self._model(self.x_valid, False, reuse=True)
            self.valid_preds = tf.argmax(logits, axis=1)
            self.valid_preds = tf.to_int32(self.valid_preds)
            self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
            self.valid_acc = tf.to_int32(self.valid_acc)
            self.valid_acc = tf.reduce_sum(self.valid_acc)

    # override
    def _build_test(self):
        print("-" * 80)
        print("Build test graph")
        logits = self._model(self.x_test, False, reuse=True)
        self.test_preds = tf.argmax(logits, axis=1)
        self.test_preds = tf.to_int32(self.test_preds)
        self.test_acc = tf.equal(self.test_preds, self.y_test)
        self.test_acc = tf.to_int32(self.test_acc)
        self.test_acc = tf.reduce_sum(self.test_acc)

    # override
    def build_valid_rl(self, shuffle=False):
        print("-" * 80)
        print("Build valid graph on shuffled data")
        with tf.device("/cpu:0"):
            # shuffled valid data: for choosing validation model
            if not shuffle and self.data_format == "NCHW":
                self.images["valid_original"] = np.transpose(
                    self.images["valid_original"], [0, 3, 1, 2])
            x_valid_shuffle, y_valid_shuffle = tf.train.shuffle_batch(
                [self.images["valid_original"], self.labels["valid_original"]],
                batch_size=self.batch_size,
                capacity=25000,
                enqueue_many=True,
                min_after_dequeue=0,
                num_threads=16,
                seed=self.seed,
                allow_smaller_final_batch=True,
            )

        logits = self._model(x_valid_shuffle, is_training=True, reuse=True)
        valid_shuffle_preds = tf.argmax(logits, axis=1)
        valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
        self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
        self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
        self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

    def connect_controller(self, controller_model):
        if self.fixed_arc is None:
            self.normal_arc, self.reduce_arc = controller_model.sample_arc
        else:
            fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
            self.normal_arc = fixed_arc[:4 * self.num_cells]
            self.reduce_arc = fixed_arc[4 * self.num_cells:]

        self._build_train()
        self._build_valid()
        self._build_test()
