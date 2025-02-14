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
from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from .. import rewards, observations, terminations
from ..components import EMPTY_POCKET_ID
from ..entities import Entities, Goal, Player
from ..states import State
from ..grid import random_positions, random_directions, room
from ..rendering.cache import RenderingCache
from .environment import Environment, Timestep
from .registry import register_env


class Room(Environment):
    random_start: bool = struct.field(pytree_node=False, default=False)

    def get_n_unique_states(self) -> int:
        n_player_states = (self.height - 2) * (self.width - 2) * 4
        n_goal_states = (self.height - 2) * (self.width - 2)
        # player and goal or just player
        if self.random_start:
            n_unique_states = n_player_states * n_goal_states
        else:
            n_unique_states = n_player_states
        return n_unique_states

    def get_unique_id_obs_fn(self) -> Array:
        """Reduces state to a unique observation id, for use with tabular Q
        learning.

        Args:
            state (State): The current state of the game.

        Returns:
            Array: An integer, where the integer uniquely identifies the observation.

        """

        def unique_id(state: State):
            # TODO: This is hardcoded for the non-random empty room environment. Find a way to
            # easily generalize to the other environments.
            full_width, full_height = state.grid.shape
            width, height = full_width - 2, full_height - 2
            n_player_states = width * height * 4
            n_free_spaces = width * height

            if self.random_start:
                goal = state.get_entity("goal")
                g_position = goal.position[0] # assuming only one goal
                g_index = jnp.asarray((g_position[0] - 1) * height + (g_position[1] - 1), dtype=jnp.int32)

                player = state.get_player()
                p_position = player.position
                p_direction = player.direction
                p_index = jnp.asarray((p_position[0] - 1) * height + (p_position[1] - 1), dtype=jnp.int32)
                p_index = (p_index * 4) + p_direction

                unique_id = (g_index * n_player_states) + p_index
                return jnp.expand_dims(unique_id, -1)
            else:
                player = state.get_player()
                position = player.position
                direction = player.direction
                index = jnp.asarray((position[0] - 1) * height + (position[1] - 1), dtype=jnp.int32)
                unique_id = index * 4 + direction
                return jnp.expand_dims(unique_id, -1)

        return unique_id

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        key, k1, k2 = jax.random.split(key, 3)

        # map
        grid = room(height=self.height, width=self.width)

        # goal and player
        if self.random_start:
            player_pos, goal_pos = random_positions(k1, grid, n=2)
            direction = random_directions(k2, n=1)
        else:
            goal_pos = jnp.asarray([self.height - 2, self.width - 2])
            player_pos = jnp.asarray([1, 1])
            direction = jnp.asarray(0)
        player = Player.create(
            position=player_pos,
            direction=direction,
            pocket=EMPTY_POCKET_ID,
        )
        # goal
        goal = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        entities = {
            Entities.PLAYER: player[None],
            Entities.GOAL: goal[None],
        }

        # systems
        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )

        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


register_env(
    "Navix-Empty-5x5-v0",
    lambda *args, **kwargs: Room.create(
        height=5,
        width=5,
        random_start=False,
        *args,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        **kwargs,
    ),
)
register_env(
    "Navix-Empty-6x6-v0",
    lambda *args, **kwargs: Room.create(
        height=6,
        width=6,
        random_start=False,
        *args,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        **kwargs,
    ),
)
register_env(
    "Navix-Empty-8x8-v0",
    lambda *args, **kwargs: Room.create(
        height=8,
        width=8,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Empty-16x16-v0",
    lambda *args, **kwargs: Room.create(
        height=16,
        width=16,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Empty-Random-5x5-v0",
    lambda *args, **kwargs: Room.create(
        height=5,
        width=5,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Empty-Random-6x6-v0",
    lambda *args, **kwargs: Room.create(
        height=6,
        width=6,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Empty-Random-8x8-v0",
    lambda *args, **kwargs: Room.create(
        height=8,
        width=8,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Empty-Random-16x16-v0",
    lambda *args, **kwargs: Room.create(
        height=16,
        width=16,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", terminations.on_goal_reached),
        *args,
        **kwargs,
    ),
)
