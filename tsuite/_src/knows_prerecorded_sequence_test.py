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
"""Tests for knows_prerecorded_sequence."""

from absl.testing import absltest
import numpy as np

from tsuite._src import agent
from tsuite._src import base
from tsuite._src import knows_prerecorded_sequence
from tsuite._src import test_utils
from tsuite._src import tsuite


SUCCESS = base.SUCCESS
FAIL = base.FAIL


class KnowsPrerecordedSequenceTest(test_utils.TSuiteTest):

  def test_knows_prerecorded_sequence_correct_behaviour(self):
    env = tsuite.TSuiteEnvironment(
        'knows_prerecorded_sequence',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays one episode with the correct behaviour.
    timestep = env.reset()
    # The following action sequence corresponds to the "secret" in the
    # knows_prerecorded_sequence.TestCase class.
    for discrete_action in [0, 3, 3, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 3, 3, 0]:
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
    self.assertEqual(timestep.reward, SUCCESS)

  def test_knows_prerecorded_sequence_incorrect_behaviour(self):
    env = tsuite.TSuiteEnvironment(
        'knows_prerecorded_sequence',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')

    # Plays one episode with the incorrect behaviour.
    timestep = env.reset()
    for discrete_action in [0, 3, 3, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 3, 3, 2]:
      action = ({'a': [np.array(discrete_action, dtype=np.int32)]},
                np.array([-0.5, 0.5], dtype=np.float32),
                np.ones((2, 3), dtype=np.float32))
      timestep = env.step(action)
    self.assertEqual(timestep.reward, FAIL)

  def test_knows_prerecorded_sequence_with_best_action(self):
    env = tsuite.TSuiteEnvironment(
        'knows_prerecorded_sequence',
        self._action_spec,
        self._observation_spec,
        default_action_name='discrete')
    # Runs 5 episodes with optimal action-sequence.
    for _ in range(5):
      timestep = env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION))
      self.assertIsNone(timestep.reward)
      while not timestep.last():
        timestep = env.step(env.read_property(tsuite.PROPERTY_BEST_ACTION))
      self.assertEqual(timestep.reward, SUCCESS)

  def test_with_agent(self):
    logs = agent.fit_agent_to_tsuite_task(
        'knows_prerecorded_sequence',
        n_updates=50)
    self.assertLess(logs[-1]['value'], 0.1)

  def test_knows_prerecorded_sequence_list_test_tasks(self):
    self.assertSetEqual(
        set(knows_prerecorded_sequence.list_test_tasks()),
        {'knows_prerecorded_sequence'})


if __name__ == '__main__':
  absltest.main()
