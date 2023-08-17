# TSuite (Test Suite)

## Overview

The intention of this environment is to provide a simple way to test RL agents
in an end-to-end setting. The tests are agnostic to the action and observation
space of the agent.

The tests are provided in the form of test-cases each containing one or more
test-tasks (think levels). The test-tasks are designed to be:

-   solvable by any reasonable agent in only a few steps,
-   compatible with most action spaces,
-   compatible with most observation spaces,
-   fast and short
-   most test-cases terminate after a single step,
-   sensitive to common mistakes (e.g. broken lstm states).

All test-tasks return a reward of 1.0 if the agent passes the test.

A good starting point to debug most systems with tsuite is the `overfit`
test-case. This test-case tests if the system can overfit to a short fixed
sequence of actions that is rewarding. This is the equivalent to the standard
debugging strategy of overfitting a small dataset in supervised learning.


## Installation

You can install the latest development version from GitHub:

```sh

python -m venv tsuite_env  # Create a virtual environment
source tsuite_env/bin/activate  # Enter the virtual environment

pip install git+https://github.com/deepmind/tsuite.git

# Run deactivate later to leave the virtual environment.
```

## User Stories

Collection of typical user stories.


### Developing and debugging an RL framework

The user develops a new RL framework e.g. training infrastructure that trains
an RL agent using using a multi-host accelerators and runs environments on a
distributed cluster (maybe akin to https://arxiv.org/abs/2104.06272).

The user wants to ensure that the frameworks surfaces and recover from errors,
handles agent states correctly and can run environments that are not
thread-safe. Relevant TSuite test-cases are:

  - `crashing_env`: will simulate an environment that crashes periodically.
  - `memory`: tests that the agent can utilize its state to memorize information
    over multiple timesteps. Importantly, the environment simulated by tsuite
    for the memory test-case is entirely stateless, so cheating is impossible.
  - `cross_contamination`: tests that no information flows across episode
    boundaries; this can happen if agent state is not correctly reset at the
    beginning and end of episodes.
  - `thread_safety`: will simulate an environment that is not thread safe in
    order to check that the RL framework isolates environments properly.
  - `bad_observation` and `bad_timestep`: will simulate an environment that
    returns bad observations and timesteps for instance NaNs and infinities.

The user can discover common implementation issues like:
corrupted agent states, incorrect resetting of agent state at episode
boundaries, silent environment crashes and/or failure to recover from
environment crashes, failure to surface or handle invalid inputs like NaNs and
infinities.


### Developing and debugging an RL agent

The user develops a new RL agent e.g. a system that perceives observations from
an environment and acts in it, and learns to maximize the reward provided by the
environment.

The user wants to ensure that the agent can perceive all provided observations
and can output all available actions. Relevant TSuite test-cases are:

  - `action_space`: tests that the agent can output the minimum and maximum
    values for the given actions.
  - `observation_space`: tests that the agent can react to information present
    in the provided observations.
  - `visual` and `language`: tests that the agent can react to information
    present in visual-like and language-like observations specifically.
  - `memory`: tests that the agent can utilize its state to memorize information
    over multiple timesteps. Importantly, the environment simulated by tsuite
    for the memory test-case is entirely stateless, so cheating is impossible.

The user can discover common implementation issues like:
off by one errors in the policy head, missing or corrupted observations,
issues in the recurrent or stateful part of the agent.


### Developing and debugging an RL algorithm

The user develops a new RL algorithm e.g. an algorithm that adapts the
parameters of an RL agent based on the reward provided by the environment.

The user wants to ensure that the algorithm can solve basic learning problems,
can learn from expert demonstrations, is sensitive to small changes in the
input. Relevant TSuite test-cases are:

  - `overfit`: tests that the algorithm can overfit to a short fixed sequence
    of actions that is rewarding. This is the equivalent to the standard
    debugging strategy of overfitting a small dataset in supervised learning.
  - `knows_prerecorded_sequence`: tests if the algorithm can overfit a long
    fixed sequence of actions. Due to the exploration problem an online RL agent
    is not expected to solve this (if it does there is probably some issue with
    evaluation), but it is possible to solve the task using expert
    demonstrations provided by tsuite.
  - `sensitivity`: tests that the agent is sensitive to numerical input at
    different scales.
  - `discount`: tests that the algorithm handles the discount provided by the
    environment correctly.
  - `causal`: tests that the algorithm discovers the correct causal structure,
    where one of the provided observations will causally influence the expected
    action, whereas another observation is only correlated.

The user can discover common issues like:
incorrect handling of discount, instabilities in the algorithm, convergence
issues, and numerical issues.


### Developing and debugging a real-time controller

The user develops a real-time controller e.g. a learned policy that controls
an environment in real-time.

The user wants to ensure that the real-time controller fulfils certain latency
guarantees, behaves correctly if the environment lags (i.e. is slow).
Relevant TSuite test-cases are:

  - `latency`: ensures that the controller provides actions to the environment
    with a maximum user-defined latency.
  - `slow_env` tests the behaviour of the controller when confronted with a
    slow environment that does violate the real-time constraints.

The user can discover common issues like: violating latency guarantees
and silent failures if guarantees from the environment side are violated.


### Developing and debugging an evaluation system

The user develops an evaluation system e.g. a visualization of the reward
obtained by an agent in real-time.

The user wants to ensure that the evaluation system works correctly i.e.
displays the information coming from the environment correctly.

Relevant TSuite test-cases are:

  - `reward`: always outputs a reward of 1, independent of the provided action.
  - `slow_env` tests the behaviour of the evaluation system if the evaluated
    environment is extremely slow.

The user can discover common issues like: the display not showing the correct
reward, timeout issues.


### Debugging custom scenarios

The user has a custom scenario that they want to debug, e.g. a specific
sequence of observations or a specific markov decision process.

The user can either implement their own TSuite test-case (see below for
a generic implementation example) or utilize the `custom` test-case, that
takes a pickle-file containing a sequence of observations and expected actions.


## Usage

### Online RL

TSuite can be used as a drop-in replacement for any dm_env compatible
environment.

The following example creates a drop-in replacement for an atari-like
environment with a single action called `ACTION` and a single observation called
`RGB`. The tsuite environment has the same action_spec and
observation_spec.

```python
import tsuite

drop_in_replacement = tsuite.TSuiteEnvironment(
    test_task='action_space@ACTION@high',
    action_spec=env_dummy.action_spec(),
    observation_spec=env_dummy.observation_spec(),
    default_action_name='ACTION',
    default_observation_name='RGB')
```

### Offline RL

In addition tsuite  can define a best, worst, and random action
at each timestep via a property.
This allows one to easily derive a dataset for each tsuite task for offline RL.

```python
import tsuite

def dataset_generator(test_task)
  drop_in_replacement = tsuite.TSuiteEnvironment(
    test_task=test_task,
    action_spec=env_dummy.action_spec(),
    observation_spec=env_dummy.observation_spec(),
    default_action_name='ACTION',
    default_observation_name='RGB')
  while True:
    action = drop_in_replacement.read_property(tsuite.PROPERTY_BEST_ACTION)
    timestep = drop_in_replacement.step(action)
    yield (timestep, action)
```


### Default Action and Observation

The user specifies a `default action` and `default observation` for each agent.
The default action is discretized into three possible values (see `Action`
class). This action is used by the agent to interact with the environment in a
standardized way, independent of the actual action-space of the agent. The
default observation is discretized into two possible values (see
`TestCase.get_observation`). These values are used to provide information to the
agent in a standardized way, independent of the actual observation-space of the
agent.


### Test Cases and Task

#### Test Task specification strings

Each test-case is defined by a TestCase class and contains one or more
test-tasks. The test-tasks are specified by a string. The string starts with
the name of the test-case, followed by an arbitrary number of arguments
separated by "@", which define the test-task.

A typical example is the ActionSpaceTestCase. An agent trained on the test-task
"action_space@action_name@high" receives a reward if it outputs the maximum
value (or a value close to the maximum in case of a continuous action) of the
action with the name "action_name".

#### Test Case Implementation

Each test-case implementation follows the same pattern:

```python
class TestCase(base.TestCase):
  """Short description of test-case.

  Long description of test-case, including what bugs this test-case is supposed
  to catch.
  """

  def __init__(self, *parameters: tuple[str], **kwargs):
    """Initializes a new TestCase.

    Args:
      *parameters: parameters specific to this test-case specified in the
        test-task specification string.
      **kwargs: additional keyword arguments forwarded to the base class.
    """
    # Initialize base-class.
    super().__init__(**kwargs)
    # Processing test-task specific parameters
    [...]

   def expected_reward_to_pass_test(self):
    """Returns the expected reward if the test-task is solved successfully."""
    # By default this is 1, if this method is overwritten it should be expressed
    # in terms of the SUCCESS and FAIL constants. E.g. if an agent is expected
    # to solve the task only half of the time, one would return 0.5 expressed
    # like this:
    return (base.SUCCESS + base.FAIL) / 2

  def reset(self) -> dm_env.TimeStep:
    # Typically the first observation returned by reset contains some kind of
    # cue for the agent, e.g. a signal (or no-signal) is injected in the default
    # observation.
    [...]
    observation = self.get_observation(signal)
    # Call the base_reset function, this is important to ensure that the
    # step_counter and episode_counter book-keeping is done correctly.
    return super().base_reset(observation=observation)

  def step(self, action) -> dm_env.TimeStep:
    # Typically the action is mapped to the internal-action so that the
    # test-task can be implemented in an action-spec agnostic way.
    internal_action = self.map_external_to_internal_action(action)
    # Determine whether the test-task is solved successfully and whether it
    # should be terminated.
    # Within the step function the two members self.episode_counter and
    # self.step_counter can be accessed,
    [...]
    # Call base_step function, this is important to ensure that the step-counter
    # and episode_counter book-keeping is done correctly.
    return super().base_step(success=success, terminate=terminate)

  def best_next_internal_action(self) -> base.InternalAction:
    """Returns the best next action based on the current state of the env."""
    [...]
    return expected_action


def list_test_tasks() -> Sequence[str]:
  """Returns available test-tasks of TestCase.

  The list_test_tasks function can take the action_spec and observation_spec
  as an argument in case this is required in order to determine valid test-task
  specification strings.
  """
  return ["test_case_name@param1@param2"]
```

### List of test-tasks

An incomplete list of possible test-tasks supported by this environment,
assuming an action space and observation space of the form:

```python
action_spec = specs.BoundedArray(
    shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='up|left')
observation_spec = specs.BoundedArray(
    shape=(4, 4, 3), dtype=np.float32, minimum=0, maximum=1, name='rgb')
```

-   Tests for the action space of the agent: action_space@up@high,
    action_space@up@low, action_space@left@high, action_space@left@low
-   Tests for the observation space of the agent: observation_space@rgb
-   Tests for the memory of the agent: memory@1, memory@2, ..., memory@N
-   Tests for behavioral cloning experiments: knows_prerecorded_sequence
-   Tests for language capabilities of the agent: language@text@content,
    language@text@length
-   Tests for the visual capabilities of the agent: visual@rgb@color,
    visual@rgb@vertical_position, visual@rgb@horizontal_position
-   Test for information leakage between episodes: cross_contamination
-   Test for information leakage from the future: future_leakage
-   Test if agent uses its discount correctly: discount@0.99.
-   Tests if agent works with zero discount: zero_discount.
-   Tests the ability of the agent to overfit to a sequence: overfit
-   Tests the ability of the agent to discover the correct causal structure:
    causal@rgb@text@90, causal@rgb@text@99
-   Tests the latency guarantee of the agent: latency@34@128
-   Tests for the agent's sensitivity to differently scaled numerical
    observations: sensitivity@rgb@-2, sensitivity@rgb@1.
-   Tests the ability of the agent to overfit to a sequence with a slow
    environment: slow_env@500
-   Tests provided by the user using a pickle-file containing episodes with
    timesteps and the expected actions: custom@/home/user/tsuite.pickle

Some tests simulate a broken environment. There is no correct behaviour in this
situation. Although, the most reasonable behaviour would be to throw a good
error message.

-   Tests for bad observations: bad_observation@rgb@nan,
    bad_observation@rgb@inf, bad_observation@rgb@dtype
-   Tests for bad timesteps: bad_timestep@step_type@oor,
    bad_timestep@discount@inf, bad_timestep@reward@nan, ...
-   Test for threading issues if the environment is not thread safe:
    thread_safety.
-   Test for behaviour if an environment crashes with a given probability:
    crashing_env@1.

A complete list can be obtained by calling the `list_test_tasks` function.

```python
import tsuite

tsuite.list_test_tasks(action_spec, observation_spec,
                       include_broken_env_tasks=True)
```

A more detailed description of the purpose and usage of each test-task can
be found in the docstring of the corresponding TestCase class.


## Technical details

### Codebase Overview

The codebase in _src/ consists of the following components:

* *tsuite.py* - which defines a dm_env environment that can be used as a drop-in
  replacement for any other dm_env environment by mocking the action and
  observation spec.
* *base.py* - defines a base-class for the test-cases, each test-case implements
  the `reset`, `step` and `best_next_internal_action` functions, and tests a
  specific aspect of an agent e.g. if the agent can memorize an input for N
  timesteps.
* most of the other files are the test-cases (action_space, observation_space,
  latency, memory, ....) - all of them follow the exact same pattern: test-case
  class implementing `__init__`, `reset`, `step` and `best_next_internal_action`
  functions; and a `list_test_tasks` function that returns some default
  test-task specification strings.
* testing infrastructure (*updater.py*) to test the test-tasks with a
  simple actor-critic agent (defined in *agent.py*).

The main complexity of the codebase is in *base.py* which defines the base-class
for the test-cases:
* it maps the action-spec of the original dm_env-like environment to a simple
  discrete 3-state action-spec (low, high and noop), so all test-cases are
  agnostic about the actual action-spec.
* it also defines a simple way to express two states (signal, no-signal) in
  the observation-spec, so that most test-cases can be agnostic to the actual
  observation-spec

On the user-side tsuite is configured using a string defining the test-task and
some parameters. e.g. "action_space@ACTION@high" would instantiate the
action_space test case for the action within the action-spec with the name
ACTION and it will test if the agent can learn to output a high value for that
action spec. What a high value means depends on the type of action-spec
e.g. for a discrete action it means the highest possible value,
for a float it means a value in the 90% quantile.


### Additional ideas for Test-Tasks

-   graph@nodes, edges, globals, senders, receivers (tests for GraphsTuple
    inputs)
-   reward_as_observation@ (tests if agent is sensitive to the reward as an
    observation)
-   stochasticity (tests if an agent can output random actions)

Please add ideas and feel free to implement them.


## Citing TSuite<a id="citing-tsuite"></a>

To cite this repository:

```
@software{tsuite2023github,
  author = {Thomas Keck},
  title = {TSuite},
  url = {http://github.com/deepmind/tsuite},
  version = {1.0.0},
  year = {2023},
}
```

In this bibtex entry, the version number is intended to be from
[`tsuite/__init__.py`](https://github.com/deepmind/tsuite/blob/main/tsuite/__init__.py),
and the year corresponds to the project's open-source release.


## License

Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Disclaimer

This is not an official Google product.
