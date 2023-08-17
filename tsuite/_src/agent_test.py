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
"""Tests for agent."""

from absl.testing import absltest
from absl.testing import parameterized
from tsuite._src import agent


class AgentTest(parameterized.TestCase):

  def test_agent(self):
    logs = agent.fit_agent_to_tsuite_task('overfit')
    self.assertGreater(logs[-1]['value'], 0.9)


if __name__ == '__main__':
  absltest.main()
