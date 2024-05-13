from abc import ABC, abstractmethod


class Model(ABC):
    """Interface for Model"""

    @abstractmethod
    def run(self, *args, **kwargs):
        """execute model"""
        pass
