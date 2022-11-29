from collections import OrderedDict, deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_env import StepType, specs


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        def default_on_none(value, default):
            if value is None:
                return default
            return value

        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=default_on_none(time_step.reward, 0.0),
            discount=default_on_none(time_step.discount, 1.0),
        )

    def specs(self):
        obs_spec = self._env.observation_spec()
        action_spec = self._env.action_spec()
        next_obs_spec = specs.Array(obs_spec.shape, obs_spec.dtype, "next_observation")
        reward_spec = specs.Array((1,), action_spec.dtype, "reward")
        discount_spec = specs.Array((1,), action_spec.dtype, "discount")
        return (obs_spec, action_spec, reward_spec, discount_spec, next_obs_spec)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="observation"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        # assert pixels_key in wrapped_obs_spec

        # pixels_shape = wrapped_obs_spec[pixels_key].shape
        pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        # pixels = time_step.observation[self._pixels_key]
        pixels = time_step.observation
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


# DM Env Wrapper for minigrid
class Minigrid(dm_env.Environment):
    def __init__(self, env):
        self.env = env
        wrapped_obs_spec = self.env.observation_space["image"]
        self._action_spec = dm_env.specs.BoundedArray(
            shape=(env.action_space.n,),
            # shape=(1,),
            dtype=np.int64,
            minimum=0,
            maximum=env.action_space.n - 1,
            name="action",
        )
        self._observation_spec = dm_env.specs.BoundedArray(
            shape=wrapped_obs_spec.shape,
            dtype=wrapped_obs_spec.dtype,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._reward_spec = dm_env.specs.Array(
            shape=(), dtype=np.float32, name="reward"
        )

    # def _transform_image(self, img):
    #     return img.transpose(2, 0, 1).copy()

    def reset(self):
        timestep = self.env.reset()
        # filter out image from tuple timestep = ({'image': [...], 'other': []}, {})
        obs = timestep[0]["image"]
        # obs = self._transform_image(obs)
        # obs = OrderedDict()
        # obs["observation"] = obs_img

        return dm_env.TimeStep(
            dm_env.StepType.FIRST, reward=None, discount=1.0, observation=obs
        )

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = obs["image"]  # take only img and drop info like direction and mission
        # obs = self._transform_image(obs)
        if done:
            return dm_env.TimeStep(
                dm_env.StepType.LAST, reward, discount=0.0, observation=obs
            )
        else:
            return dm_env.TimeStep(
                dm_env.StepType.MID, reward, discount=1.0, observation=obs
            )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render()

    def seed(self, seed=None):
        return self.env.seed(seed)


def make_env(name):
    suite, task = name.split("-", 1)
    print(suite)
    if suite == "MiniGrid":  # name_format = "MiniGrid-Empty-8x8-v0"
        from minigrid.wrappers import (RGBImgObsWrapper,
                                       RGBImgPartialObsWrapper, gym)

        # env = Minigrid(name)
        env = gym.make(name, render_mode="rgb_array")
        # env = RGBImgPartialObsWrapper(env)
        env = RGBImgObsWrapper(env)
        print(env.render_mode)
        env = Minigrid(env)
        env = ExtendedTimeStepWrapper(env)
        env = FrameStackWrapper(env, 3)
        print("env created")
    else:
        env = None

    return env
