"""Common environment wrappers for RL algorithms."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from gymnasium.spaces import Box

class FrameStack(gym.Wrapper):
    """Stack n_frames last frames."""
    
    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = None
        
        wrapped_obs_shape = env.observation_space.shape
        low = np.repeat(env.observation_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], n_frames, axis=0)
        self.observation_space = Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self.frames = np.tile(obs, (self.n_frames, 1, 1))
        return self.frames, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = obs
        return self.frames, reward, terminated, truncated, info

class EpisodeMonitor(gym.Wrapper):
    """Keep track of episode statistics."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_returns = []
        self.episode_lengths = []
        self._current_return = 0.0
        self._current_length = 0
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._current_return = 0.0
        self._current_length = 0
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_return += reward
        self._current_length += 1
        
        if terminated or truncated:
            info["episode"] = {
                "r": self._current_return,
                "l": self._current_length
            }
            self.episode_returns.append(self._current_return)
            self.episode_lengths.append(self._current_length)
        
        return obs, reward, terminated, truncated, info

class NormalizeObservation(gym.Wrapper):
    """Normalize observations to range [0, 1]."""
    
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        obs_space = env.observation_space
        self.obs_rms_mean = np.zeros(obs_space.shape, dtype=np.float32)
        self.obs_rms_var = np.ones(obs_space.shape, dtype=np.float32)
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info
    
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_rms_mean) / np.sqrt(self.obs_rms_var + self.epsilon)
    
    def update_normalization(self, obs_batch: np.ndarray) -> None:
        """Update running mean and variance."""
        batch_mean = np.mean(obs_batch, axis=0)
        batch_var = np.var(obs_batch, axis=0)
        batch_count = obs_batch.shape[0]
        
        delta = batch_mean - self.obs_rms_mean
        tot_count = batch_count + 1
        
        new_mean = self.obs_rms_mean + delta * batch_count / tot_count
        m_a = self.obs_rms_var * 1
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.obs_rms_mean = new_mean
        self.obs_rms_var = new_var 