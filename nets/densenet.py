# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains model definitions for versions of the Oxford VGG network.

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

@@densenet_40
@@densenet_100
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def densenet_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def composite_function(_input, out_features, is_training = True, dropout_keep_prob = 0.8, kernel_size = [3,3]):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
        # BN
        output = slim.batch_norm(_input, is_training=is_training)#!!need op
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = slim.conv2d(output, out_features, kernel_size)
        # dropout(in case of training and in case it is no 1.0)
        if is_training:
            output = slim.dropout(output, dropout_keep_prob)
    return output

def bottleneck(_input, out_features, is_training = True, dropout_keep_prob = 0.8, kernel_size = [1,1]):
    with tf.variable_scope("bottleneck"):
        output = slim.batch_norm(_input, is_training=is_training)
        output = tf.nn.relu(output)
        inter_features = out_features * 4
        output = slim.conv2d(output, inter_features, kernel_size, padding='VALID')
        if is_training:
            output = slim.dropout(output, dropout_keep_prob)
    return output
       
def add_internal_layer(_input, growth_rate, bc_mode):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not bc_mode:
            comp_out = composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif bc_mode:
            bottleneck_out = bottleneck(_input, out_features=growth_rate)
            comp_out = composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        
        # concatenate _input with out from composite function
        # the only diffenence between resnet and densenet
        output = tf.concat(axis=3, values=(_input, comp_out))

def transition_layer(_input, is_training = True, dropout_keep_prob = 1.0, reduction = 1.0):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # call composite function with 1x1 kernel
    out_features = int(int(_input.get_shape()[-1]) * reduction)
    output = composite_function(#!! need dropout??
        _input, out_features, is_training, dropout_keep_prob, kernel_size=[1,1])
    # run average pooling
    output = slim.avg_pool2d(output, [2,2])
    return output        

def trainsition_layer_to_classes(_input, n_classes = 1001, is_training = True):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = output = slim.batch_norm(_input, is_training=is_training)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = slim.avg_pool2d(output, [last_pool_kernel, last_pool_kernel])
        # FC
        logits = slim.fully_connected(output, n_classes, activation_fn = None, normalizer_fn = None)
        return logits
        
def densenet_40(inputs,
            first_output_features,
            layers_per_block,
            growth_rate,
            num_classes=1001,
            is_training=True,
            dropout_keep_prob=0.8,
            scope='densenet_40',
            fc_conv_padding='VALID'):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'densenet_40', [inputs]) as sc:
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    end_points = {}
    with slim.arg_scope([slim.conv2d, slim.max_pool2d]):
        
        #first conv
        net = slim.conv2d(inputs, first_output_features, [3,3])
        end_points['conv1'] = net
        
        #block1
        with tf.variable_scope("Block_1"):
            for layer in range(layers_per_block):
                with tf.variable_scope("layer_%d" % layer):
                    net = add_internal_layer(net, growth_rate)
            with tf.variable_scope("Transition_after_block_1"):
                    net = transition_layer(net)
        end_points['block1'] = net
        
        #block2
        with tf.variable_scope("Block_2"):
            for layer in range(layers_per_block):
                with tf.variable_scope("layer_%d" % layer):
                    net = add_internal_layer(net, growth_rate)
            with tf.variable_scope("Transition_after_block_2"):
                    net = transition_layer(net)
        end_points['block2'] = net
                    
        #block3
        with tf.variable_scope("Block_3"):
            for layer in range(layers_per_block):
                with tf.variable_scope("layer_%d" % layer):
                    net = add_internal_layer(net, growth_rate)
            with tf.variable_scope("trainsition_layer_to_classes"):
                net = trainsition_layer_to_classes(net, num_classes)
        end_points['block3'] = net
        
        prediction = tf.nn.softmax(logits)
    
    return prediction, end_points
  


def densenet_100():
    pass