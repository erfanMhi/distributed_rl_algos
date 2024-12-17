
from abc import ABC, abstractmethod
from typing import Callable, Dict

import gymnasium as gym


class BaseRLAlgorithm(ABC):
    """Base class for all RL algorithms."""
    
    def __init__(self, env_func: Callable[[], gym.Env], cfg: Dict):
        self.env_func = env_func
        self.cfg = cfg

    @abstractmethod
    def learn(self) -> None:
        """Main learning loop for the algorithm."""
        raise NotImplementedError("Subclass must implement learn() method")
