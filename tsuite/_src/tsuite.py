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
"""Unittest Reinforcement Learning environment."""

from collections.abc import Sequence, Mapping
import copy

from typing import Any, Optional

from absl import logging

import dm_env
import numpy as np
import tree


from tsuite._src import action_space
from tsuite._src import bad_observation
from tsuite._src import bad_timestep
from tsuite._src import base
from tsuite._src import causal
from tsuite._src import crashing_env
from tsuite._src import cross_contamination
from tsuite._src import custom
from tsuite._src import discount
from tsuite._src import future_leakage
from tsuite._src import knows_prerecorded_sequence
from tsuite._src import language
from tsuite._src import latency
from tsuite._src import memory
from tsuite._src import observation_space
from tsuite._src import overfit
from tsuite._src import reward
from tsuite._src import sensitivity
from tsuite._src import slow_env
from tsuite._src import thread_safety
from tsuite._src import visual
from tsuite._src import zero_discount


PROPERTY_BEST_ACTION = "best_action"
PROPERTY_WORST_ACTION = "worst_action"
PROPERTY_RANDOM_ACTION = "random_action"


def list_test_tasks(
    action_spec: base.ActionSpecsTree,
    observation_spec: base.SpecsTree,
    include_broken_env_tasks: bool = True,
) -> Sequence[str]:
  """Returns a list of all available test-tasks."""
  action_spec = base.set_names_in_spec(action_spec)
  observation_spec = base.set_names_in_spec(observation_spec)
  tasks = [
      *action_space.list_test_tasks(action_spec),
      *discount.list_test_tasks(),
      *causal.list_test_tasks(observation_spec),
      *cross_contamination.list_test_tasks(),
      *custom.list_test_tasks(),
      *future_leakage.list_test_tasks(),
      *knows_prerecorded_sequence.list_test_tasks(),
      *latency.list_test_tasks(),
      *language.list_test_tasks(observation_spec),
      *memory.list_test_tasks(),
      *observation_space.list_test_tasks(observation_spec),
      *overfit.list_test_tasks(),
      *reward.list_test_tasks(),
      *sensitivity.list_test_tasks(observation_spec),
      *slow_env.list_test_tasks(),
      *visual.list_test_tasks(observation_spec),
      *zero_discount.list_test_tasks(),
  ]
  # The following tasks simulate a broken environment. It is reasonable for an
  # agent to crash with a meaningful error message for these test tasks.
  if include_broken_env_tasks:
    tasks += [
        *bad_observation.list_test_tasks(observation_spec),
        *bad_timestep.list_test_tasks(),
        *crashing_env.list_test_tasks(),
        *thread_safety.list_test_tasks(),
    ]
  return tasks


class TSuiteEnvironment(dm_env.Environment):
  """A test environment built on the dm_env.Environment class.

  Supported test-cases are:
    - action_space: tests the action space of the agent, see
      `action_space.TestCase`.
    - bad_observation: tests for bad observations, see
      `bad_observation.TestCase`.
    - bad_timestep: tests for bad timesteps, see `bad_timestep.TestCase`.
    - causal: test that an agent discovers correct causal relationship.
    - crashing_env: tests behaviour on environment errors,
      see `crashing_env.TestCase`.
    - cross_contamination: checks for cross contamination between episodes, see
      `cross_contamination.TestCase`.
    - custom: tests user-provided custom test-cases, see `custom.TestCase`.
    - discount: tests that an agent uses its discount correctly, see
      `discount.TestCase`.
    - future_leakage: checks for violations in causality, see
      `future_leakage.TestCase`.
    - knows_prerecorded_sequence: tests the ability to learn a prerecorded
      sequence e.g. from expert demonstrations, see
      `knows_prerecorded_sequence.TestCase`.
    - latency: tests the latency guarantees of the agent,
      see `latency.TestCase`.
    - language: tests the language capabilities of the agent, see
      `language.TestCase`.
    - memory: tests the memory of the agent, see `memory.TestCase`.
    - observation_space: tests the observation space of the agent, see
      `observation_space.TestCase`.
    - overfit: tests the ability of the agent to overfit to a sequence, see
      `overfit.TestCase`.
    - reward: tests the reward processing, see `reward.TestCase`.
    - sensitivity: tests the agent's sensitivity to differently scaled numerical
      observations, see `latency.TestCase`.
    - slow_env: tests the ability of the agent to overfit to a sequence from a
      slow environment, see `slow_env.TestCase`.
    - thread_safety: test for threading issues if the environment is not thread
      safe, see `thread_safety.TestCase`.
    - visual: tests the visual capabilities of the agent, see
      `visual.TestCase`.
    - zero_discount: ests if agent works with zero discount, see
      `zero_discount.TestCase`.

  Additional arguments required by the test-cases are encoded in the test-task
  name and are separated by the '@' character.

  For instance,
    - to test if the agent can output the maximum value of a discrete action
      named "discrete", the agent should be tested on the test task with the
      name "action_space@discrete@high".
    - to test if the agent can retain provided information for 5 environment
      steps, the agent should be tested on the test task with the name
      "memory@5".
  """

  def __init__(self,
               test_task: str,
               action_spec: base.ActionSpecsTree,
               observation_spec: base.SpecsTree,
               default_action_name: Optional[str] = None,
               default_observation_name: Optional[str] = None,
               validate_action_spec: bool = True,
               remove_nones: bool = False,
               verbose_logging: bool = False):
    """Initializes a new unittest environment.

    Args:
      test_task: name of the test task.
      action_spec: defines the action space of the agent.
      observation_spec: defines the observations consumed by the agent.
      default_action_name: name of the action in the action_spec, which is
        used to check for a reaction from the agent by default.
      default_observation_name: name of the observation in the observation_spec,
        which is used to provide signals to the agent by default.
      validate_action_spec: whether to validate the given action in the step
        function against the action_spec.
      remove_nones: whether to remove None values from the reward and discount
        in the first timestep.
      verbose_logging: whether to log additional information like actions,
        timesteps, etc. This can be useful for debugging purposes.
    """
    logging.info("Unittest test task %s", test_task)
    logging.info("Unittest ActionSpec %s", action_spec)
    logging.info("Unittest ObservationSpec %s", observation_spec)
    logging.info("Unittest default action name %s", default_action_name)
    logging.info("Unittest default observation name %s",
                 default_observation_name)

    self._action_spec = copy.deepcopy(action_spec)
    self._observation_spec = copy.deepcopy(observation_spec)
    self._reset_next_step = True
    self._validate_action_spec = validate_action_spec
    self._remove_nones = remove_nones
    self._verbose_logging = verbose_logging

    impl_cls: Mapping[str, base.TestCaseCtor] = {
        "action_space": action_space.TestCase,
        "bad_observation": bad_observation.TestCase,
        "bad_timestep": bad_timestep.TestCase,
        "causal": causal.TestCase,
        "crashing_env": crashing_env.TestCase,
        "cross_contamination": cross_contamination.TestCase,
        "custom": custom.TestCase,
        "discount": discount.TestCase,
        "future_leakage": future_leakage.TestCase,
        "knows_prerecorded_sequence": knows_prerecorded_sequence.TestCase,
        "latency": latency.TestCase,
        "language": language.TestCase,
        "memory": memory.TestCase,
        "observation_space": observation_space.TestCase,
        "overfit": overfit.TestCase,
        "reward": reward.TestCase,
        "sensitivity": sensitivity.TestCase,
        "slow_env": slow_env.TestCase,
        "thread_safety": thread_safety.TestCase,
        "visual": visual.TestCase,
        "zero_discount": zero_discount.TestCase,
    }
    case, *args = test_task.split("@")
    self._impl = impl_cls[case](
        *args,
        action_spec=self._action_spec,
        observation_spec=self._observation_spec,
        default_action_name=default_action_name,
        default_observation_name=default_observation_name)

  def read_property(self, key: str) -> Any:
    self._verbose_log(f"Called read_property with key {key}.")
    if key == PROPERTY_BEST_ACTION:
      value = self._impl.map_internal_to_external_action(
          self._impl.best_next_internal_action())
    elif key == PROPERTY_WORST_ACTION:
      internal_action = self._impl.best_next_internal_action()
      if internal_action == base.InternalAction.LOW:
        internal_action = base.InternalAction.HIGH
      elif internal_action == base.InternalAction.HIGH:
        internal_action = base.InternalAction.LOW
      # Note: Leave action == base.Action.NOOP as it is.
      value = self._impl.map_internal_to_external_action(internal_action)
    elif key == PROPERTY_RANDOM_ACTION:
      def random_action_from_spec(spec) -> base.ArrayTree:
        if np.issubdtype(spec.dtype, np.integer):
          return np.random.randint(
              spec.minimum, spec.maximum + 1, dtype=spec.dtype)
        elif np.issubdtype(spec.dtype, np.inexact):
          return np.random.uniform(
              spec.minimum, spec.maximum).astype(spec.dtype)
        else:
          raise ValueError(f"Unsupported dtype {spec.dtype} for action spec.")
      value = tree.map_structure(random_action_from_spec, self._action_spec)
    else:
      value = ""
    self._verbose_log(f"Finished read_property returning {value}.")
    return value

  def _verbose_log(self, log: str):
    if self._verbose_logging:
      logging.info(log)

  def write_property(self, key: str, value: str) -> None:
    logging.warning("Attempting to write property %s: %s", key, value)
    pass

  def _maybe_remove_none(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    if self._remove_nones and timestep.first():
      if timestep.reward is None:
        self._verbose_log("Replacing None reward with 0.0.")
        timestep = timestep._replace(reward=0.0)
      if timestep.discount is None:
        self._verbose_log("Replacing None discount with 1.0.")
        timestep = timestep._replace(discount=1.0)
    return timestep

  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    self._verbose_log("Called reset.")
    self._reset_next_step = False
    ts = copy.deepcopy(self._maybe_remove_none(self._impl.reset()))
    self._verbose_log(f"Finished reset returning timestep {ts}.")
    return ts

  def step(self, action: base.ArrayTree) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    self._verbose_log(f"Called step with action {action}.")
    if self._reset_next_step:
      ts = self.reset()
      self._verbose_log(f"Finished step returning timestep {ts}.")
      return ts

    if self._validate_action_spec:
      self._verbose_log("Validating action.")
      tree.map_structure(
          lambda leaf_spec, leaf_action: leaf_spec.validate(leaf_action),
          self._action_spec, action)
    transition = self._impl.step(action)
    if transition.last():
      self._reset_next_step = True
    ts = copy.deepcopy(self._maybe_remove_none(transition))
    self._verbose_log(f"Finished step returning timestep {ts}.")
    return ts

  def observation_spec(self) -> base.SpecsTree:
    """Returns the observation spec."""
    self._verbose_log("Called observation_spec.")
    spec = copy.deepcopy(self._observation_spec)
    self._verbose_log(f"Finished observation_spec returning spec {spec}.")
    return spec

  def action_spec(self) -> base.SpecsTree:
    """Returns the action spec."""
    self._verbose_log("Called action_spec.")
    spec = copy.deepcopy(self._action_spec)
    self._verbose_log(f"Finished action_spec returning spec {spec}.")
    return spec
