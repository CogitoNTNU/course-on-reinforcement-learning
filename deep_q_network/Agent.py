from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def save(self, path):
        raise NotImplemented

    @abstractmethod
    def load(self, path):
        raise NotImplemented

    @abstractmethod
    def act(self, state):
        raise NotImplemented
