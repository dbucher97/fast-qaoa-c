import os
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

PROBLEM_DATA_PATH = "data/"
PROBLEM_DATA_PATH = os.environ.get("PROBLEM_DATA_PATH", PROBLEM_DATA_PATH)
_instances = {}


@dataclass
class ProblemBase(ABC):
    id: int
    n_qubits: int

    @abstractmethod
    def random_instance(cls, *args, id=None, **kwargs):
        ...

    @classmethod
    def get_instances(cls):
        global _instances
        if not cls.__name__ in _instances:
            path = Path(__file__).parent.parent / PROBLEM_DATA_PATH
            path = path / f"{cls.__name__}.feather"
            if path.exists():
                data = {}
                df = pd.read_feather(path)
                def inner(r):
                    ks = cls(**dict(r))
                    if ks.n_qubits not in data:
                        data[ks.n_qubits] = []
                    data[ks.n_qubits].append(ks)

                df.apply(inner, axis=1)
            else:
                data = {}

            _instances[cls.__name__] = data

        return _instances[cls.__name__]

    @classmethod
    def next_id(cls, n_qubits: int):
        store = cls.get_instances()
        if n_qubits not in store or len(store[n_qubits]) == 0:
            return 0
        else:
            return store[n_qubits][-1].id + 1

    @classmethod
    def store(cls):
        store = cls.get_instances()
        if len(store) == 0:
            return
        path = os.path.join(PROBLEM_DATA_PATH, f"{cls.__name__}.feather")
        data = []
        for _, v in store.items():
            data += [list(asdict(instance).values()) for instance in v]
        columns = list(asdict(store[next(iter(store.keys()))][0]).keys())
        df = pd.DataFrame(data, columns=columns)
        df.to_feather(path)

    def add(self):
        store = self.__class__.get_instances()
        if self.n_qubits not in store:
            store[self.n_qubits] = []
        store[self.n_qubits].append(self)
        return self

    @classmethod
    def clear(cls):
        cls.get_instances().clear()
        path = os.path.join(PROBLEM_DATA_PATH, f"{cls.__name__}.feather")
        if os.path.exists(path):
            os.remove(path)


    @abstractmethod
    def decache(self):
        ...
