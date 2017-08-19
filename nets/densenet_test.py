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
"""Tests for densenet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import densenet

slim = tf.contrib.slim


class DensenetTest(tf.test.TestCase):

    def testBuild(self):
        batch_size = 5
        height, width = 32, 32
        num_classes = 10
        first_output_features = 24
        layers_per_block = 12
        growth_rate = 12
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            logits, _ = densenet.densenet_40(inputs, first_output_features, layers_per_block, growth_rate)
            self.assertEquals(logits.op.name, 'densenet_40/Softmax')
            self.assertListEqual(logits.get_shape().as_list(),
                                 [batch_size, num_classes])
 
    def testEndPoints(self):
        batch_size = 5
        height, width = 32, 32
        num_classes = 10
        first_output_features = 24
        layers_per_block = 12
        growth_rate = 12
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            _, end_points = densenet.densenet_40(inputs, first_output_features, layers_per_block, growth_rate)
            expected_names = ['densenet_40/first_conv/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_1/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_2/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_3/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_4/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_5/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_6/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_7/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_8/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_9/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_10/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_11/composite_function/Conv', 
                              'densenet_40/block_1/Repeat/add_internal_layer_12/composite_function/Conv', 
                              'densenet_40/block_1/transition_1/composite_function/Conv', 
                              'densenet_40/block_1/transition_1/AvgPool2D', 
                              'densenet_40/block_2/Repeat/add_internal_layer_1/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_2/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_3/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_4/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_5/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_6/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_7/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_8/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_9/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_10/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_11/composite_function/Conv', 
                              'densenet_40/block_2/Repeat/add_internal_layer_12/composite_function/Conv', 
                              'densenet_40/block_2/transition_2/composite_function/Conv', 
                              'densenet_40/block_2/transition_2/AvgPool2D', 
                              'densenet_40/block_3/Repeat/add_internal_layer_1/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_2/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_3/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_4/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_5/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_6/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_7/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_8/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_9/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_10/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_11/composite_function/Conv', 
                              'densenet_40/block_3/Repeat/add_internal_layer_12/composite_function/Conv', 
                              'densenet_40/block_3/trainsition_layer_to_classes/AvgPool2D', 
                              'densenet_40/block_3/trainsition_layer_to_classes/fully_connected']
            self.assertSetEqual(set(end_points.keys()), set(expected_names))


if __name__ == '__main__':
  tf.test.main()
