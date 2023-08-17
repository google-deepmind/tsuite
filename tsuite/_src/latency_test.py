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
"""Tests for latency."""

import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tsuite._src import base
from tsuite._src import latency
from tsuite._src import test_utils
from tsuite._src import tsuite

SUCCESS = base.SUCCESS
FAIL = base.FAIL


class LatencyTest(test_utils.TSuiteTest):

  def test_input_validation(self):
    with self.subTest('LatencyMustBePositive'):
      with self.assertRaises(ValueError):
        tsuite.TSuiteEnvironment(
            'latency@-1@5',
            self._action_spec, self._observation_spec,
            default_action_name='discrete',
            default_observation_name='float')

    with self.subTest('LatencyCantBeZero'):
      with self.assertRaises(ValueError):
        tsuite.TSuiteEnvironment(
            'latency@0@5',
            self._action_spec, self._observation_spec,
            default_action_name='discrete',
            default_observation_name='float')

    with self.subTest('EpisodeLengthMustBePositive'):
      with self.assertRaises(ValueError):
        tsuite.TSuiteEnvironment(
            'latency@10@-1',
            self._action_spec, self._observation_spec,
            default_action_name='discrete',
            default_observation_name='float')

    with self.subTest('EpisodeLengthCantBeZero'):
      with self.assertRaises(ValueError):
        tsuite.TSuiteEnvironment(
            'latency@10@0',
            self._action_spec, self._observation_spec,
            default_action_name='discrete',
            default_observation_name='float')

  @parameterized.parameters(
      (100, 1),
      (200, 1),
      (100, 5),
      (200, 5),
  )
  def test_latency_success(self, latency_in_ms, episode_length):
    env = tsuite.TSuiteEnvironment(
        f'latency@{latency_in_ms}@{episode_length}',
        self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    timestep = env.reset()
    self.assertTrue(timestep.first())
    action = ({'a': [np.array(0, dtype=np.int32)]},
              np.array([-0.5, 0.5], dtype=np.float32),
              np.ones((2, 3), dtype=np.float32))
    for i in range(episode_length):
      timestep = env.step(action)
      if i != episode_length - 1:
        self.assertEqual(timestep.reward, 0)
        self.assertTrue(timestep.mid())
    self.assertEqual(timestep.reward, SUCCESS)
    self.assertTrue(timestep.last())

  @parameterized.parameters(
      (100, 1),
      (200, 1),
      (100, 5),
      (200, 5),
  )
  def test_latency_failure(self, latency_in_ms, episode_length):
    env = tsuite.TSuiteEnvironment(
        f'latency@{latency_in_ms}@{episode_length}',
        self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    action = ({'a': [np.array(0, dtype=np.int32)]},
              np.array([-0.5, 0.5], dtype=np.float32),
              np.ones((2, 3), dtype=np.float32))

    with self.subTest('FirstStepSlow'):
      timestep = env.reset()
      # Wait until latency constraint is violated.
      time.sleep(latency_in_ms / 1000.)
      self.assertTrue(timestep.first())
      for i in range(episode_length):
        timestep = env.step(action)
        if i != episode_length - 1:
          self.assertEqual(timestep.reward, 0)
          self.assertTrue(timestep.mid())
      self.assertEqual(timestep.reward, FAIL)
      self.assertTrue(timestep.last())

    with self.subTest('LastStepSlow'):
      timestep = env.reset()
      self.assertTrue(timestep.first())
      for i in range(episode_length):
        if i == episode_length - 1:
          # Wait until latency constraint is violated.
          time.sleep(latency_in_ms / 1000.)
        timestep = env.step(action)
        if i != episode_length - 1:
          self.assertEqual(timestep.reward, 0)
          self.assertTrue(timestep.mid())
      self.assertEqual(timestep.reward, FAIL)
      self.assertTrue(timestep.last())

  def test_latency_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'latency@100@1', self._action_spec, self._observation_spec,
        default_action_name='discrete',
        default_observation_name='float')
    # Runs 5 episodes with optimal action-sequence.
    for _ in range(5):
      self.assertIsNone(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward)
      self.assertEqual(
          env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION)).reward,
          SUCCESS)

  def test_reward_list_test_tasks(self):
    self.assertSetEqual(
        set(latency.list_test_tasks()),
        {'latency@10@128', 'latency@34@128', 'latency@125@128'})


if __name__ == '__main__':
  absltest.main()
