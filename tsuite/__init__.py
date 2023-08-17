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
"""Tsuite: Get your RL agent fixed today!"""


from tsuite._src.tsuite import list_test_tasks
from tsuite._src.tsuite import PROPERTY_BEST_ACTION
from tsuite._src.tsuite import PROPERTY_RANDOM_ACTION
from tsuite._src.tsuite import PROPERTY_WORST_ACTION
from tsuite._src.tsuite import TSuiteEnvironment

from tsuite._src.updater import mini_batch_generator
from tsuite._src.updater import Updater


__version__ = "1.0"

__all__ = (
    "list_test_tasks",
    "mini_batch_generator",
    "Updater",
    "PROPERTY_BEST_ACTION",
    "PROPERTY_RANDOM_ACTION",
    "PROPERTY_WORST_ACTION",
    "TSuiteEnvironment",
    )


#  _________________________________________
# / Please don't use symbols in `_src`. They \
# \ are not part of the Tsuite public API.   /
#  ------------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
