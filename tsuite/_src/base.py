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
"""Basic definitions for the tsuite environment."""

from collections.abc import Callable, Iterable, Mapping, Sequence
import enum
import hashlib

from typing import Any, Optional, Union, Protocol

from absl import logging
import dm_env
from dm_env import specs
import numpy as np
import tree


ArrayTree = Union[np.ndarray, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
SpecsTree = Union[specs.Array, Iterable["SpecsTree"], Mapping[Any, "SpecsTree"]]

ActionSpecType = Union[specs.BoundedArray, specs.StringArray]
ActionSpecsTree = Union[
    ActionSpecType,
    Iterable["ActionSpecsTree"],
    Mapping[Any, "ActionSpecsTree"]]


class InternalAction(enum.Enum):
  """Discretized action space used by the test-cases.

  All test-cases assume that the agent is able to communicate three distinct
  states with their default action.

  See `TestCase.get_action` for details on how discrete and continuous actions
  are mapped to these three states.
  """
  LOW, NOOP, HIGH = list(range(3))


class ExternalStringAction(str, enum.Enum):
  """External actions of type string that match to the internal actions.

  The strings have been manually chosen to return 0, 1, 2, when mapped by the
  external_to_internal_action mapping method of the TestCase below.
  """
  LOW = "red"
  NOOP = "green"
  HIGH = "blue"


# Rewards returned by TestCase for success and fail.
SUCCESS = 1.0
FAIL = 0.0


# Minimum and maximum values for observations.
OBSERVATION_MAX_DEFAULT = 1
OBSERVATION_MIN_DEFAULT = 0


def _fingerprint(x: Union[str, np.ndarray, bytes]) -> int:
  """Returns a fingerpint of a string."""
  if isinstance(x, np.ndarray):
    x = x.item()
  if isinstance(x, str):
    x = x.encode("utf-8")
  assert isinstance(x, bytes), f"Expected bytes got {type(x)} with value {x}"
  return int(hashlib.sha256(x).hexdigest(), 16)


def set_names_in_spec(nested_spec: SpecsTree) -> SpecsTree:
  """Returns spec with None replaced by a proper name.

  If the name of an element in a nested spec is None, it is replaced
  with the string representation of the path of this element in the nest.

  Args:
    nested_spec: nested spec.
  """
  def visitor_function(path, element: specs.Array) -> specs.Array:
    if element.name is None:
      if isinstance(element, specs.BoundedArray):
        element = specs.BoundedArray(
            shape=element.shape,
            dtype=element.dtype,
            maximum=element.maximum,
            minimum=element.minimum,
            name="_".join(map(str, path)))
      else:
        element = specs.Array(
            shape=element.shape,
            dtype=element.dtype,
            name="_".join(map(str, path)))
    return element

  return tree.map_structure_with_path(visitor_function, nested_spec)


def make_signal_injector_visitor_fn(
    observation_names: Sequence[str],
) -> Callable[[specs.Array], np.ndarray]:
  """Returns function that injects signal into the given observations."""
  def visitor_function(node: specs.Array) -> np.ndarray:
    if str(node.name) in observation_names:
      if node.dtype.type in [np.bytes_, np.str_]:
        # Returns an array of bytes/strings filled with '1'.
        return np.full(node.shape, OBSERVATION_MAX_DEFAULT, node.dtype)
      else:
        value = OBSERVATION_MAX_DEFAULT
        if hasattr(node, "maximum"):
          value = node.maximum
        return np.asarray(np.full(node.shape, value, node.dtype), node.dtype)
    else:
      if node.dtype.type in [np.bytes_, np.str_]:
        # Returns an array of bytes/strings filled with '0'.
        return np.full(node.shape, OBSERVATION_MIN_DEFAULT, node.dtype)
      else:
        value = OBSERVATION_MIN_DEFAULT
        if hasattr(node, "minimum"):
          value = node.minimum
        return np.asarray(np.full(node.shape, value, node.dtype), node.dtype)
  return visitor_function


def _extract_scalar_bounds_spec_with_name(
    element: specs.Array, index: tuple[int], name: str
) -> specs.BoundedArray | specs.StringArray:
  """Returns a spec with scalar bounds and the given name."""
  if isinstance(element, specs.BoundedArray):
    # The rest of the code assumes that maximum and minimum are single
    # values. A new BoundedArray spec is created with the maximum and
    # minimum of the selected action.
    if element.maximum.shape:
      maximum = element.maximum[index]
    else:
      maximum = element.maximum
    if element.minimum.shape:
      minimum = element.minimum[index]
    else:
      minimum = element.minimum
    element = specs.BoundedArray(
        shape=element.shape,
        dtype=element.dtype,
        maximum=maximum,
        minimum=minimum,
        name=name)
  elif element.dtype == bool:
    element = specs.BoundedArray(
        shape=element.shape,
        dtype=element.dtype,
        maximum=True,
        minimum=False,
        name=name)
  elif isinstance(element, specs.StringArray):
    pass
  else:
    raise ValueError(f"Unsupported action spec {element}.")
  return element


class TestCaseCtor(Protocol):
  """Protocol for the test-case implementations used by tsuite."""

  def __call__(self,
               *args: str,
               action_spec: ActionSpecsTree,
               observation_spec: SpecsTree,
               default_action_name: Optional[str] = None,
               default_observation_name: Optional[str] = None,
               verbose_logging: bool = False):
    """Initializes a new TestCase.

    Args:
      *args: string arguments passed to the TestCase constructor.
      action_spec: defines the action space of the agent.
      observation_spec: defines the observations consumed by the agent.
      default_action_name: name of the action in the action_spec, which is
        used to check for a reaction from the agent by default.
        If None, the first action in the action spec is used as default action.
      default_observation_name: name of the observation in the observation_spec,
        which is used to provide signals to the agent by default.
        If None, the first observation in the observation spec is used as
        default observation.
      verbose_logging: whether to log additional information. This can be useful
        for debugging purposes.
    """


class TestCase():
  """Base class for the test-case implementations used by the unittest env."""

  def __init__(self,
               *,
               action_spec: ActionSpecsTree,
               observation_spec: SpecsTree,
               default_action_name: Optional[str] = None,
               default_observation_name: Optional[str] = None,
               verbose_logging: bool = False):
    """Initializes a new TestCase.

    Args:
      action_spec: defines the action space of the agent.
      observation_spec: defines the observations consumed by the agent.
      default_action_name: name of the action in the action_spec, which is
        used to check for a reaction from the agent by default.
        If None, the first action in the action spec is used as default action.
      default_observation_name: name of the observation in the observation_spec,
        which is used to provide signals to the agent by default.
        If None, the first observation in the observation spec is used as
        default observation.
      verbose_logging: whether to log additional information. This can be useful
        for debugging purposes.
    """
    self._action_spec = set_names_in_spec(action_spec)
    self._default_action_name = default_action_name
    self._observation_spec = set_names_in_spec(observation_spec)
    self._default_observation_name = default_observation_name
    self._verbose_logging = verbose_logging

    if self._default_action_name is None:
      self._default_action_name = tree.flatten(self._action_spec)[0].name
      # Ensures action name is represented as a string.
      # It is supported that the action-spec contains non-string names, but
      # internally we represent and match everything based on the string
      # representation
      self._default_action_name = str(self._default_action_name)
    self._default_action_path, self._default_action_spec = (
        self._parse_action_spec())
    if not isinstance(self._default_action_spec, ActionSpecType):
      raise ValueError(f"Unsupported action spec {self._default_action_spec}.")

    if self._default_observation_name is None:
      self._default_observation_name = tree.flatten(observation_spec)[0].name
      # Ensures observation name is represented as a string.
      # It is supported that the action-spec contains non-string names, but
      # internally we represent and match everything based on the string
      # representation
      self._default_observation_name = str(self._default_observation_name)

    self._default_observation_path, self._default_observation_spec = (
        self._parse_observation_spec())

    self.step_counter = 0
    self.episode_counter = 0

  def _verbose_log(self, log: str):
    if self._verbose_logging:
      logging.info(log)

  def expected_reward_to_pass_test(self) -> float:
    """Returns the expected reward required to pass the test.

    Most tests require the agent to solve the given task and obtain a reward
    of 1.0. On the other hand, some tests check if an agent learns in situations
    where it should not learn anything. The expected reward for those tests is
    usually 0.5.
    """
    return SUCCESS

  def _parse_action_spec(
      self) -> tuple[tuple[Any, ...], ActionSpecType]:
    """Returns the path and spec of the default action."""
    self._verbose_log(f"Called _parse_action_spec for {self._action_spec}")
    for path, element in tree.flatten_with_path(self._action_spec):
      self._verbose_log(f"Processing {path}, {element}.")
      n_dimensions = len(element.shape)
      # Ensures element name is represented as a string.
      # It is supported that the action-spec contains non-string names, but
      # internally we represent and match everything based on the string
      # representation
      name = str(element.name)
      if n_dimensions == 0:
        self._verbose_log(f"{name} is dimension 0.")
        if self._default_action_name == name:
          self._verbose_log(
              f"Finished _parse_action_spec returning {path}, {element}.")
          return path, element
      else:
        # Assumes that name of a tensor-like action, encodes the names of the
        # sub-action in the form "action1|action2|...". The gotham environment
        # fullfills this assumption.
        if "|" in name:
          names = name.split("|")
        else:
          names = [name + f"_{index}"
                   for index in range(np.prod(element.shape))]
        self._verbose_log(
            f"{name} is dimension {n_dimensions} and was parsed to {names}.")
        for index, name in enumerate(names):
          self._verbose_log(f"Inner processing {index}, {name}.")
          if self._default_action_name == name:
            if index >= np.prod(element.shape):
              raise ValueError(
                  "ActionSpec name indicates more elements than the shape!")
            index = np.unravel_index(index, element.shape)
            element = _extract_scalar_bounds_spec_with_name(
                element, index, name)
            path = path + index
            self._verbose_log(
                f"Finished _parse_action_spec returning {path}, {element}.")
            return path, element
    else:
      raise ValueError(
          f"Could not find element named {self._default_action_name} "
          f"in {self._action_spec}")

  def _parse_observation_spec(self) -> tuple[tuple[Any, ...], specs.Array]:
    """Returns the path and spec of the default observation."""
    self._verbose_log(
        f"Called _parse_observation_spec for {self._observation_spec}")
    for path, element in tree.flatten_with_path(self._observation_spec):
      if self._default_observation_name == str(element.name):
        self._verbose_log(
            f"Finished _parse_observation_spec returning {path}, {element}.")
        return path, element
    else:
      raise ValueError(
          f"Could not find element named {self._default_observation_name} "
          f"in {self._observation_spec}")

  def base_reset(
      self, observation: Optional[ArrayTree] = None) -> dm_env.TimeStep:
    """Returns timestep and resets the internal state of this test case.

    This function should be used by subclasses to ensure that the episodes
    and the steps are correctly counted.

    Args:
      observation: returned observation.
    """
    self._verbose_log(f"Called base_reset with {observation}.")
    self._verbose_log(
        f"State step: {self.step_counter} episode: {self.episode_counter}.")
    self.step_counter = 0
    self.episode_counter += 1
    if observation is None:
      observation = self.get_observation()
    ts = dm_env.restart(observation)
    self._verbose_log(f"Finished base_reset returning {ts}.")
    return ts

  def reset(self) -> dm_env.TimeStep:
    return self.base_reset()

  def base_step(
      self,
      terminate: bool = False,
      success: bool = False,
      observation: Optional[ArrayTree] = None) -> dm_env.TimeStep:
    """Returns timestep.

    This function should be used by subclasses to ensure that the steps are
    correctly counted.

    Args:
      terminate: whether to terminate the episode.
      success: whether to return a reward or not.
      observation: returned observation.
    """
    self._verbose_log(
        f"Called base_reset with {terminate} {success} {observation}.")
    self._verbose_log(
        f"State step: {self.step_counter} episode: {self.episode_counter}.")
    self.step_counter += 1
    if observation is None:
      observation = self.get_observation()
    reward = SUCCESS if success else FAIL
    if terminate:
      ts = dm_env.termination(reward=reward, observation=observation)
    else:
      ts = dm_env.transition(reward=reward, observation=observation)
    self._verbose_log(f"Finished base_step returning {ts}.")
    return ts

  def step(self, action: ArrayTree) -> dm_env.TimeStep:
    del action
    return self.base_step()

  def best_next_internal_action(self) -> InternalAction:
    """Returns the best next action based on the current state of the env."""
    self._verbose_log("Called best_next_internal_action returning NOOP.")
    return InternalAction.NOOP

  def map_external_to_internal_action(
      self, external_action: ArrayTree) -> InternalAction:
    """Returns the action corresponding to the default action name.

    Actions are discretized into LOW, NOOP, HIGH.
    - If the default action is discrete, LOW corresponds to the minimum value,
      HIGH to the maximum value, and NOOP to all other values.
    - If the default action is continuous, LOW corresponds to a value of 10%,
      HIGH to a value of 90% and NOOP to all other values, where the percent
      is defined with a linear equation between the minimum (0%) and
      maximum (100%) value of the action.

    Args:
      external_action: np.array with the structure defined by the action_spec.
    """
    self._verbose_log(
        f"Called map_external_to_internal_action with {external_action}.")
    # Traverses the action structure to extract the desired default action.
    for action_path in self._default_action_path:
      external_action = external_action[action_path]

    if (isinstance(self._default_action_spec, specs.BoundedArray) and
        (np.issubdtype(self._default_action_spec.dtype, np.integer) or
         self._default_action_spec.dtype == bool)):
      if external_action == self._default_action_spec.maximum:
        internal_action = InternalAction.HIGH
      elif external_action == self._default_action_spec.minimum:
        internal_action = InternalAction.LOW
      else:
        internal_action = InternalAction.NOOP
    elif (isinstance(self._default_action_spec, specs.BoundedArray) and
          np.issubdtype(self._default_action_spec.dtype, np.inexact)):
      def linear_from_spec(x, spec):
        return x * (spec.maximum - spec.minimum) + spec.minimum
      if external_action > linear_from_spec(0.9, self._default_action_spec):
        internal_action = InternalAction.HIGH
      elif external_action < linear_from_spec(0.1, self._default_action_spec):
        internal_action = InternalAction.LOW
      else:
        internal_action = InternalAction.NOOP
    elif isinstance(self._default_action_spec, specs.StringArray):
      internal_action = InternalAction(_fingerprint(external_action) % 3)
    else:
      raise ValueError(
          f"Unsupported dtype {self._default_action_spec.dtype} "
          "for action spec.")
    self._verbose_log(
        f"Finished map_external_to_internal_action returning {internal_action}."
    )
    return internal_action

  def map_internal_to_external_action(
      self, internal_action: InternalAction) -> ArrayTree:
    """Returns action in original action-space.

    Args:
      internal_action: Discretized action used by tsuite internally.
    """
    self._verbose_log(
        f"Called map_internal_to_external_action with {internal_action}.")
    def _get_external_action(path, spec: specs.BoundedArray) -> ArrayTree:
      array = spec.generate_value()
      # The default_action_path includes the indices of the array for
      # vector-like arrays. Hence, we only check for the same prefix here.
      prefix = self._default_action_path[:len(path)]
      postfix = self._default_action_path[len(path):]
      if path == prefix:
        if internal_action == InternalAction.NOOP:
          override = _spec_noop_value(spec)
        elif internal_action == InternalAction.LOW:
          override = _spec_min_max_value(spec)[0]
        elif internal_action == InternalAction.HIGH:
          override = _spec_min_max_value(spec)[1]
        else:
          raise ValueError(f"Unknown action: {internal_action}")
        if hasattr(override, "shape") and override.shape == array.shape:
          array = override
        else:
          array[tuple(postfix)] = override
      return array

    external_action = tree.map_structure_with_path(
        _get_external_action, self._action_spec)
    self._verbose_log(
        f"Finished map_internal_to_external_action returning {external_action}."
    )
    return external_action

  def get_observation(self, signal: bool = False) -> ArrayTree:
    """Returns observation with the structure defined by the observation_spec.

    The observation consists of the minimum value for each element in the
    observation spec. An additional signal can be present, represented by the
    maximum value for the element with the default observation name.

    Args:
      signal: whether to inject a signal into the default observation.
    """
    signal_obs = []
    if signal:
      signal_obs.append(self._default_observation_name)
    return tree.map_structure(
        make_signal_injector_visitor_fn(signal_obs), self._observation_spec)


def _spec_min_max_value(spec: specs.Array):
  if isinstance(spec, specs.BoundedArray):
    return (spec.minimum, spec.maximum)
  elif isinstance(spec, specs.StringArray):
    return (ExternalStringAction.LOW.value, ExternalStringAction.HIGH.value)
  elif spec.dtype == bool:
    return (False, True)
  else:
    iinfo = np.iinfo(spec.dtype)
    return (iinfo.min, iinfo.max)


def _spec_noop_value(spec: specs.Array):
  """Returns noop value for external action given its spec."""
  if np.issubdtype(spec.dtype, np.integer):
    min_value, max_value = _spec_min_max_value(spec)
    return np.asarray((min_value + max_value) // 2).astype(spec.dtype)
  elif np.issubdtype(spec.dtype, np.inexact):
    min_value, max_value = _spec_min_max_value(spec)
    return np.asarray((min_value + max_value) / 2.0).astype(spec.dtype)
  elif isinstance(spec, specs.StringArray):
    return np.asarray(
        ExternalStringAction.NOOP.value).astype(spec.dtype)
  elif spec.dtype == bool:
    return False
  else:
    raise ValueError(f"Unsupported dtype {spec.dtype} for action spec.")
