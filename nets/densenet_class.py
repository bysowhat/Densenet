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

slim = tf.contrib.slim

# =========================================================================== #
# Densenet class definition.
# =========================================================================== #
DensenetParams = namedtuple('DensenetParameters', ['num_classes',
                                         'first_output_features',
                                         'layers_per_block',
                                         'growth_rate',
                                         'bc_mode',
                                         'is_training'
                                         ])

class DENSENet(object):
    """Implementation of the Densenet network.

    The default features layers are:
      conv1 ==> !!!!
      block1 ==> 19 x 19
      block2 ==> 10 x 10
      block3 ==> 5 x 5
    The default image size used to train this network is !!!
    """
    default_params = DensenetParams(
        num_classes = 1001,
        first_output_features = 4,
        layers_per_block = 12,
        growth_rate = 12,
        bc_mode = False,
        is_training = True,
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
            dropout_keep_prob=0.8,
            scope='densenet_40'):
        """Densenet network definition.
        """
        r = self.densenet_40(inputs,
                    first_output_features = self.params.first_output_features,
                    layers_per_block = self.params.layers_per_block,
                    growth_rate = self.params.growth_rate,
                    num_classes = self.params.num_classes,
                    is_training = self.params.is_training,
                    dropout_keep_prob = dropout_keep_prob,
                    scope = scope)
        return r
    
    def densenet_arg_scope(weight_decay=0.0005):#!!where?
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
    
    def composite_function(self, _input, out_features, is_training = self.is_training, dropout_keep_prob = 0.8, kernel_size = [3,3]):
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
            output = slim.conv2d(output, out_features, kernel_size)#!bn act?
            # dropout(in case of training and in case it is no 1.0)
            if is_training:
                output = slim.dropout(output, dropout_keep_prob)
        return output
    
    def bottleneck(self, _input, out_features, is_training = self.is_training, dropout_keep_prob = 0.8, kernel_size = [1,1]):
        with tf.variable_scope("bottleneck"):
            output = slim.batch_norm(_input, is_training=is_training)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = slim.conv2d(output, inter_features, kernel_size, padding='VALID')
            if is_training:
                output = slim.dropout(output, dropout_keep_prob)
        return output
           
    def add_internal_layer(self, _input, growth_rate):
            """Perform H_l composite function for the layer and after concatenate
            input with output from composite function.
            """
            # call composite function with 3x3 kernel
            if not self.bc_mode:
                comp_out = self.composite_function(
                    _input, out_features=growth_rate, kernel_size=3)
            elif self.bc_mode:
                bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
                comp_out = self.composite_function(
                    bottleneck_out, out_features=growth_rate, kernel_size=3)
            
            # concatenate _input with out from composite function
            # the only diffenence between resnet and densenet
            output = tf.concat(axis=3, values=(_input, comp_out))
    
    def transition_layer(self, _input, is_training = self.is_training, dropout_keep_prob = 1.0, reduction = 1.0):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * reduction)
        output = self.composite_function(#!! need dropout??
            _input, out_features, is_training, dropout_keep_prob, kernel_size=[1,1])
        # run average pooling
        output = slim.avg_pool2d(output, [2,2])
        return output        
    
    def trainsition_layer_to_classes(self, _input, n_classes = 1001, is_training = self.is_training):
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
    
    def densenet_40(self, inputs,
            first_output_features,
            layers_per_block,
            growth_rate,
            num_classes=1001,
            is_training=True,
            dropout_keep_prob=0.8,
            scope='densenet_40'):
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
                            net = self.add_internal_layer(net, growth_rate)
                    with tf.variable_scope("Transition_after_block_1"):
                            net = self.transition_layer(net)
                end_points['block1'] = net
                
                #block2
                with tf.variable_scope("Block_2"):
                    for layer in range(layers_per_block):
                        with tf.variable_scope("layer_%d" % layer):
                            net = self.add_internal_layer(net, growth_rate)
                    with tf.variable_scope("Transition_after_block_2"):
                            net = self.transition_layer(net)
                end_points['block2'] = net
                            
                #block3
                with tf.variable_scope("Block_3"):
                    for layer in range(layers_per_block):
                        with tf.variable_scope("layer_%d" % layer):
                            net = self.add_internal_layer(net, growth_rate)
                    with tf.variable_scope("trainsition_layer_to_classes"):
                        net = self.trainsition_layer_to_classes(net, num_classes)
                end_points['block3'] = net
                
                prediction = tf.nn.softmax(logits)
            
            return prediction, end_points



    #ssd!
    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    def losses():
        pass
    








 


