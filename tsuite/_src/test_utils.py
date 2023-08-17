# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests utils."""

from absl.testing import parameterized
from dm_env import specs
import numpy as np


class TSuiteTest(parameterized.TestCase):
  """Helper class which sets up action and obs spec."""

  def setUp(self):
    super().setUp()
    self._action_spec = (
        {'a': [specs.DiscreteArray(num_values=4, name='discrete')]},
        specs.BoundedArray(
            shape=(2,), dtype=np.float32, minimum=[-1, -1], maximum=1,
            name='cont_up|cont_left'),
        specs.BoundedArray(
            shape=(2, 3),
            dtype=np.float32,
            minimum=np.zeros((2, 3)),
            maximum=np.ones((2, 3)),
            name='tensor',
        ),
    )
    self._observation_spec = {
        'rgb': specs.BoundedArray(shape=(8, 8, 3), dtype=np.float32, minimum=-1,
                                  maximum=1, name='rgb'),
        'text': (specs.BoundedArray(shape=(5,), dtype=np.int32, minimum=0,
                                    maximum=9, name='text'),
                 specs.DiscreteArray(num_values=5, name='text_length')),
        'raw_text': specs.Array((), dtype=np.str_, name='raw_text'),
        'dvector': specs.BoundedArray(shape=(3,), dtype=np.int32,
                                      minimum=-1, maximum=1, name='dvector'),
        'float': specs.BoundedArray(shape=(), minimum=-1, maximum=1,
                                    dtype=np.float32, name='float'),
    }
