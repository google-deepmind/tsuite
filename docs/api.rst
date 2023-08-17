API
===

.. currentmodule:: tsuite

.. autosummary::

    list_test_tasks
    mini_batch_generator
    Updater
    PROPERTY_BEST_ACTION
    PROPERTY_RANDOM_ACTION
    PROPERTY_WORST_ACTION
    TSuiteEnvironment


Tsuite Environment
~~~~~~~~~~~~~~~~~~

.. autofunction:: list_test_tasks
.. autoclass:: TSuiteEnvironment


Properties (for Offline RL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodata::  PROPERTY_BEST_ACTION
.. autodata::  PROPERTY_RANDOM_ACTION
.. autodata::  PROPERTY_WORST_ACTION


Learning Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction::  mini_batch_generator
.. autoclass::  Updater
