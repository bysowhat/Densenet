# Copyright 2017 bysowhat. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Densenet network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple

slim = tf.contrib.slim

# =========================================================================== #
# Densenet class definition.
# =========================================================================== #
DensenetParams = namedtuple('DensenetParameters', ['num_classes',
                                         'first_output_features',
                                         'layers_per_block',
                                         'growth_rate',
                                         'bc_mode',
                                         'is_training',
                                         'dropout_keep_prob'
                                         ])

class DENSENet(object):
    """Implementation of the Densenet network.

   
    """
    default_params = DensenetParams(
        num_classes = 10,
        first_output_features = 24,
        layers_per_block = 12,
        growth_rate = 12,
        bc_mode = False,
        is_training = True,
        dropout_keep_prob = 0.8,
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, DensenetParams):
            self.params = params
        else:
            self.params = DENSENet.default_params

    # ======================================================================= #
    def net(self, inputs,
            scope='densenet_40'):
        """Densenet network definition.
        """
        r = densenet_40(inputs,
                    first_output_features = self.params.first_output_features,
                    layers_per_block = self.params.layers_per_block,
                    growth_rate = self.params.growth_rate,
                    num_classes = self.params.num_classes,
                    is_training = self.params.is_training,
                    bc_mode = self.params.bc_mode,
                    dropout_keep_prob = self.params.dropout_keep_prob,
                    scope = scope)
        return r
    
    def arg_scope(self, weight_decay=0.0004, is_training = True, data_format='NHWC'):
        """Network arg_scope.
        """
        return densenet_arg_scope(weight_decay, is_training, data_format)

    def losses():
        pass
    


def densenet_arg_scope(weight_decay=0.0004, is_training = True, data_format='NHWC'):
    """Defines the Densenet arg scope.
    
    Args:
      weight_decay: The l2 regularization coefficient.
      is_training: for batch_norm

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=None,
                        normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:
            return sc
            

def composite_function(_input, out_features, training = True, dropout_keep_prob = 0.8, kernel_size = [3,3]):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
        # BN
        output = slim.batch_norm(_input, is_training=training)#!!need op
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = slim.conv2d(output, out_features, kernel_size)
        # dropout(in case of training and in case it is no 1.0)
        if training:
            output = slim.dropout(output, dropout_keep_prob)
    return output

def bottleneck(_input, out_features, training = True, dropout_keep_prob = 0.8):
    with tf.variable_scope("bottleneck"):
        inter_features = out_features * 4
        output = slim.batch_norm(_input, is_training=training)#!!need op
        output = tf.nn.relu(output)
        output = slim.conv2d(_input, inter_features, [1,1], padding='VALID')
        if is_training:
            output = slim.dropout(output, dropout_keep_prob)
    return output
       
def add_internal_layer(_input, growth_rate, training = True, bc_mode = False, dropout_keep_prob = 1.0):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not bc_mode:
            _output = composite_function(_input, growth_rate, training)
            if training:
                _output = slim.dropout(_output, dropout_keep_prob)
                
        elif bc_mode:
            bottleneck_out = bottleneck(_input, growth_rate, is_training)
            _output = composite_function(bottleneck_out, growth_rate, training)
            if training:
                _output = slim.dropout(_output, dropout_keep_prob)
        
        # concatenate _input with out from composite function
        # the only diffenence between resnet and densenet
        output = tf.concat(axis=3, values=(_input, _output))
        return output

def transition_layer(_input, num_filter, training = True, dropout_keep_prob = 0.8, reduction = 1.0):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # call composite function with 1x1 kernel
    num_filter = int(num_filter * reduction)
    _output = composite_function(_input, num_filter, training, kernel_size = [1,1])
    if training:
        _output = slim.dropout(_output, dropout_keep_prob)
    _output = slim.avg_pool2d(_output, [2,2])
    return _output        

def trainsition_layer_to_classes(_input, n_classes = 10, training = True):
    """This is last transition to get probabilities by classes. It perform:
    - batch normalization
    - ReLU nonlinearity
    - wide average pooling
    - FC layer multiplication
    """
    _output = output = slim.batch_norm(_input, is_training=training)
    #L.scale in caffe
    _output = tf.nn.relu(_output)
    last_pool_kernel = int(_output.get_shape()[-2])
    _output = slim.avg_pool2d(_output, [last_pool_kernel, last_pool_kernel])
    logits = slim.fully_connected(_output, n_classes)
    return logits


def densenet_40(inputs,
        first_output_features,
        layers_per_block,
        growth_rate,
        num_classes=10,
        is_training=True,
        bc_mode = False,
        dropout_keep_prob=0.8,
        scope='densenet_40'):
    """Densenet -Layers version 40 without bc struct Example.
       The default features layers are:
          conv1 ==> 32 x 32 x 4
          block1 ==> 32 x 32 x 16
          transition1 ==> 16 x 16 x 16
          block2 ==> 16 x 16 x 28
          transition2 ==> 8 x 8 x 28
          block3 ==> 8 x 8 x 40
          transition3 ==> 1 x 1 x 40
    
    Note: To use in classification mode, resize input to 32x32(cifar).
    
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
          layers during training.
        scope: Optional scope for the variables.
    
      Returns:
        the last op containing the log predictions and end_points dict.
    """
    nchannels = first_output_features
    with tf.variable_scope(scope, 'densenet_40', [inputs]) as sc:
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        end_points = {}
        #first conv
        net = slim.conv2d(inputs, first_output_features, [3,3], scope="conv1")
        end_points['conv1'] = net
        
        #block1
        with tf.variable_scope("Block_1"):
            for layer in range(layers_per_block):
                with tf.variable_scope("layer_%d" % layer):
                    net = add_internal_layer(net, growth_rate, is_training, bc_mode, dropout_keep_prob)
                    nchannels += growth_rate
            with tf.variable_scope("Transition_1"):
                    net = transition_layer(net, nchannels, is_training)
        end_points['block1'] = net
        
        #block2
        with tf.variable_scope("Block_2"):
            for layer in range(layers_per_block):
                with tf.variable_scope("layer_%d" % layer):
                    net = add_internal_layer(net, growth_rate, is_training, bc_mode, dropout_keep_prob)
                    nchannels += growth_rate
            with tf.variable_scope("Transition_2"):
                    net = transition_layer(net, nchannels, is_training)
        end_points['block2'] = net
        
        #block3
        with tf.variable_scope("Block_3"):
            for layer in range(layers_per_block):
                with tf.variable_scope("layer_%d" % layer):
                    net = add_internal_layer(net, growth_rate, is_training, bc_mode, dropout_keep_prob)
                    nchannels += growth_rate
                    
            with tf.variable_scope("trainsition_layer_to_classes"):
                net = trainsition_layer_to_classes(net, num_classes, is_training)
        end_points['block3'] = net
        
        #softmax
        net = tf.reshape(net, [-1,num_classes])
        prediction = tf.nn.softmax(net)
        
        return prediction, end_points



 


