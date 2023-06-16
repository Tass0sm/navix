# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict

import jax
from jax import Array
from flax import struct
from jax.random import KeyArray
import jax.numpy as jnp


class Component(struct.PyTreeNode):
    """A component is a part of the state of the environment."""
    position: Array = jnp.zeros((1, 2), dtype=jnp.int32) - 1


# class HasId(Component):
#     """A component that has an id"""

#     id: Array = jnp.asarray(0)


# class HasPosition(Component):
#     """A component that has a position in the environment"""

#     position: Array = jnp.asarray(0)


# class HasDirection(Component):
#     """A component that has a direction"""

#     direction: Array = jnp.asarray(0)


# class Stochastic(Component):
#     """A component that has a probability"""

#     # key: KeyArray
#     probability: Array = jnp.asarray(1.0)


# class Holder(Component):
#     """A component wiht a 1-slot pocket to hold other components (id)"""

#     pocket: Array = jnp.asarray(0)


# class Replaceable(Component):
#     """A component that can be replaced by another component (id)"""

#     replacement: Array = jnp.asarray(0)


# class Consumable(Component):
#     """A component that can be consumed by another component (id)"""

#     requires: Array = jnp.asarray(0)


class Player(Component):
    """Players are entities that can act around the environment"""

    direction: Array = jnp.asarray(0)
    pocket: Array = jnp.asarray(0)


class Goal(Component):
    """Goals are entities that can be reached by the player"""

    probability: Array = jnp.ones((1,), dtype=jnp.float32)

class Pickable(Component):
    """Pickable items are world objects that can be picked up by the player.
    Examples of pickable items are keys, coins, etc."""

    id: Array = jnp.asarray(-1)


class Consumable(Component):
    """Consumable items are world objects that can be consumed by the player.
    Consuming an item requires a tool (e.g. a key to open a door).
    A tool is an id (int) of another item, specified in the `requires` field (-1 if no tool is required).
    After an item is consumed, it is both removed from the `state.entities` collection, and replaced in the grid
    by the item specified in the `replacement` field (0 = floor by default).
    Examples of consumables are doors (to open) food (to eat) and water (to drink), etc.
    """
    requires: Array = jnp.zeros((1,), dtype=jnp.int32) - 1
    replacement: Array = jnp.zeros((1,), dtype=jnp.float32)


class State(struct.PyTreeNode):
    """The Markovian state of the environment"""

    key: KeyArray
    """The random number generator state"""
    grid: Array
    """The 2D-grid containing the ids of the entities in each position"""
    player: Player  # we can potentially extend this to multiple players easily
    """The player entity"""
    goals: Goal = Goal()
    """The goal entity, batched over the number of goals"""
    keys: Pickable = Pickable()
    """The key entity, batched over the number of keys"""
    doors: Consumable = Consumable()
    """The door entity, batched over the number of doors"""


class StepType(IntEnum):
    TRANSITION = 0
    """discount > 0, episode continues"""
    TRUNCATION = 1
    """discount > 0, episode ends"""
    TERMINATION = 2
    """discount == 0, episode ends"""


class Timestep(struct.PyTreeNode):
    t: Array
    observation: Array
    action: Array
    reward: Array
    step_type: Array
    state: State
    info: Dict[str, Any] = struct.field(default_factory=dict)
