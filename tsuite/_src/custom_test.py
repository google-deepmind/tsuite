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
"""Tests for custom."""


import os
import pickle
import tempfile

from absl.testing import absltest
import numpy as np

from tsuite._src import base
from tsuite._src import custom
from tsuite._src import test_utils
from tsuite._src import tsuite


SUCCESS = base.SUCCESS
FAIL = base.FAIL


def _get_episodes():
  red_array = np.zeros((8, 8, 3))
  red_array[..., 0] = 1
  green_array = np.zeros((8, 8, 3))
  green_array[..., 1] = 1
  blue_array = np.zeros((8, 8, 3))
  blue_array[..., 2] = 1
  return [
      [
          {
              'timestep': dict(observation={'rgb': red_array}),
              'external_action': ({'a': [np.array(0)]},)
          },
          {},  # Use default tsuite timestep for last-timestep.
      ],
      [
          {
              'timestep': dict(observation={'rgb': green_array}),
              'external_action': ({'a': [np.array(1)]},)
          },
          {},  # Use default tsuite timestep for last-timestep.
      ],
      [
          {
              'timestep': dict(observation={'rgb': blue_array}),
              'external_action': ({'a': [np.array(3)]},)
          },
          {},  # Use default tsuite timestep for last-timestep.
      ]
  ]


class CustomTest(test_utils.TSuiteTest):

  def setUp(self):
    super().setUp()
    fd, self._path = tempfile.mkstemp(suffix='.pickle')
    os.close(fd)
    with open(self._path, 'wb') as f:
      pickle.dump(_get_episodes(), f)

  def tearDown(self):
    try:
      os.remove(self._path)
    except OSError:
      pass
    super().tearDown()

  def test_custom(self):
    env = tsuite.TSuiteEnvironment(
        f'custom@{self._path}',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')

    with self.subTest('red_episode_correct'):
      timestep = env.reset()
      np.testing.assert_array_almost_equal(
          timestep.observation['rgb'].mean(axis=0).mean(axis=0),
          np.array([1.0, 0.0, 0.0]))
      action = ({'a': [np.array(0, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

    with self.subTest('green_episode_correct'):
      timestep = env.reset()
      np.testing.assert_array_almost_equal(
          timestep.observation['rgb'].mean(axis=0).mean(axis=0),
          np.array([0.0, 1.0, 0.0]))
      action = ({'a': [np.array(1, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

    with self.subTest('blue_episode_correct'):
      timestep = env.reset()
      np.testing.assert_array_almost_equal(
          timestep.observation['rgb'].mean(axis=0).mean(axis=0),
          np.array([0.0, 0.0, 1.0]))
      action = ({'a': [np.array(3, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, SUCCESS)

    with self.subTest('red_episode_incorrect'):
      timestep = env.reset()
      np.testing.assert_array_almost_equal(
          timestep.observation['rgb'].mean(axis=0).mean(axis=0),
          np.array([1.0, 0.0, 0.0]))
      action = ({'a': [np.array(1, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, FAIL)

    with self.subTest('green_episode_incorrect'):
      timestep = env.reset()
      np.testing.assert_array_almost_equal(
          timestep.observation['rgb'].mean(axis=0).mean(axis=0),
          np.array([0.0, 1.0, 0.0]))
      action = ({'a': [np.array(3, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, FAIL)

    with self.subTest('blue_episode_incorrect'):
      timestep = env.reset()
      np.testing.assert_array_almost_equal(
          timestep.observation['rgb'].mean(axis=0).mean(axis=0),
          np.array([0.0, 0.0, 1.0]))
      action = ({'a': [np.array(0, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
      self.assertEqual(timestep.reward, FAIL)

  def test_custom_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        f'custom@{self._path}',
        self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    # Runs 5 episodes with optimal action-sequence.
    for _ in range(5):
      self.assertIsNone(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward)
      self.assertEqual(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward,
          SUCCESS)

  def test_custom_list_test_tasks(self):
    self.assertEmpty(custom.list_test_tasks())


if __name__ == '__main__':
  absltest.main()
