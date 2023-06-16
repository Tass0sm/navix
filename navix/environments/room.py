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

from typing import Dict

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jax.typing import ArrayLike

from navix.components import Component

from ..components import Goal, Player, State, Timestep
from ..grid import room, spawn_entity
from .environment import Environment


class Room(Environment):
    def reset(self, key: KeyArray) -> Timestep:
        key, k1, k2 = jax.random.split(key, 3)

        # player
        player = Player(position=jnp.asarray([0, 0]))
        # goal
        goal = Goal(position=jnp.asarray([self.width - 1, self.height - 1]))
        # map
        grid = room(self.width, self.height)

        # systems
        state = State(
            key=key,
            grid=grid,
            player=player,
            goals=goal,
        )

        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation(state),
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )
