import os
from typing import List, Optional, Union, Dict, Any, TypeVar, Tuple
from pydantic.dataclasses import dataclass
from dataclasses import asdict
from problems import ALL_PROBLEMS
from enum import Enum
from pprint import pformat
import pandas as pd

T = TypeVar("T")


def aslist(x: Union[T, List[T]]) -> List[T]:
    if not isinstance(x, list):
        return [x]
    else:
        return x


class InitialConditions(str, Enum):
    Constant = "constant"
    Linear = "linear"
    Random = "random"
    GridSearch = "grid_search"


class QAOAKind(str, Enum):
    MaskedCost = "masked_cost"
    QuadPenaltyCost = "quad_penalty_cost"
    QuadPenaltyFullProblem = "quad_penalty_full_problem"
    QPE = "qpe"


@dataclass
class Settings:
    ...


@dataclass
class MaskedCostSettings(Settings):
    ...


@dataclass
class QuadPenaltyCostSettings(Settings):
    penalty: Optional[float] = None


@dataclass
class QuadPenaltyFullProblemSettings(QuadPenaltyCostSettings):
    ...


@dataclass
class QPESettings(Settings):
    ancilla: Union[float, List[float]]
    shift: Union[float, List[float]] = 0


class CostKind(str, Enum):
    Default = "default"
    Mask = "masked_cost"


@dataclass
class Experiment:
    kind: QAOAKind
    cost: CostKind = CostKind.Default
    settings: Optional[Dict[str, Any]] = None
    until_size: Optional[int] = None
    settings_obj: Optional[Settings] = None

    def __repr__(self):
        rem = 80 - 4 - len(self.kind.name)
        res = f"-- {self.kind.name} " + "-" * rem + "\n"
        res += f"Cost function = {self.cost.name}\n"
        if self.until_size is not None:
            res += f"Until size    = {self.until_size}\n"
        if self.settings_obj is not None:
            res += pformat(asdict(self.settings_obj)) + "\n"
        return res


ProblemsKind = Enum("ProblemsKind", [(p.__name__, p.__name__) for p in ALL_PROBLEMS])


@dataclass
class ExperimentCollection:
    name: str
    problem: ProblemsKind
    sizes: Union[int, List[int]]

    depths: Union[int, List[int]]
    qaoa: Union[Experiment, List[Experiment]]

    initial: InitialConditions = InitialConditions.Constant

    instances: Optional[int] = None

    beta_scale: float = 1
    gamma_scale: float = 1

    interpolate: bool = True
    repeat: int = 1

    path: str = "./results/"
    _result = None

    class Config:
        exclude = ["_result"]

    def get_stored(self, query):
        if self._result is None:
            return None
        else:
            res = self._result
            for k, v in query.items():
                if v is None:
                    res = res[res[k].isna()]
                else:
                    res = res[res[k] == v]
                if len(res) == 0:
                    return None
            return res.iloc[0]

    def load_results(self):
        path = os.path.join(self.path, self.name + ".feather")
        if os.path.exists(path):
            self._result = pd.read_feather(path)
            return self._result
        else:
            return None

    def add_results(self, df: pd.DataFrame):
        path = os.path.join(self.path, self.name + ".feather")
        os.makedirs(self.path, exist_ok=True)
        self._result = pd.concat([self._result, df], ignore_index=True)
        self._result.to_feather(path)

    def __repr__(self):
        start = 4 * " "
        res = ""
        res += f"Experiment Collection '{self.name}'\n"
        res += f"{start}Problem    = {self.problem.name}\n"
        res += f"{start}Instances  = {'all' if self.instances is None else self.instances}\n"
        res += f"{start}Sizes      = {aslist(self.sizes)}\n"
        res += (
            f"{start}Parameters = {self.initial.name}("
            + f"β={self.beta_scale}, γ={self.gamma_scale}"
            + (", interpolate" if self.interpolate else "")
            + ")\n"
        )
        res += f"{start}Depths     = {aslist(self.depths)}\n"

        return "\n".join([res] + list(map(repr, self.qaoa)))
