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

import densenet_class as densenet

slim = tf.contrib.slim


class DensenetTest(tf.test.TestCase):

  def testBuild(self):
    batch_size = 5
    height, width = 32, 32
    num_classes = 10
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = densenet.densenet_40(inputs, num_classes)
      self.assertEquals(logits.op.name, 'vgg_a/fc8/squeezed')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

 


if __name__ == '__main__':
  tf.test.main()
