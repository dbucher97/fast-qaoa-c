from typing import List, Optional, Dict, Any
from pydantic.dataclasses import dataclass
import toml
import sys
from enum import Enum
from problems import ALL_PROBLEMS
from problems.problem import ProblemBase

from fastqaoa import params


class InitialConditions(str, Enum):
    Constant = "constant"
    Linear = "linear"
    Random = "random"


class QAOAKind(str, Enum):
    Vanilla = "vanilla"
    Mask = "mask"
    QuadPenalty = "quad_penalty"
    QuadPenaltyFullProblem = "quad_penalty_full_problem"


class CostKind(str, Enum):
    Default = "default"
    Mask = "mask"


@dataclass
class QAOARun:
    kind: QAOAKind
    cost: CostKind = CostKind.Default
    settings: Optional[Dict[str, Any]] = None
    until_size: Optional[int] = None


ProblemsKind = Enum("ProblemsKind", [(p.__name__, p.__name__) for p in ALL_PROBLEMS])


@dataclass
class QAOAExperiment:
    name: str
    problem: ProblemsKind
    sizes: List[int]

    depths: List[int]
    qaoa: List[QAOARun]

    inital: InitialConditions = InitialConditions.Constant

    instances: Optional[int] = None

    beta_scale: float = 0.1
    gamma_scale: float = 0.1

    interpolate: bool = True
    repeat: int = 1


def get_initial(exp: QAOAExperiment, depth):
    if exp.inital == InitialConditions.Constant:
        betas, gammas = params.init_const(depth)
    elif exp.inital == InitialConditions.Linear:
        betas, gammas = params.init_linear(depth)
    elif exp.params == InitialConditions.Random:
        betas, gammas = params.init_random(depth)
    betas *= exp.beta_scale
    gammas *= exp.gamma_scale
    return betas, gammas

def run_qaoa_interpolate(instance: ProblemBase, qaoa_run: QAOARun, exp: QAOAExperiment):
    betas, gammas = get_initial(exp, exp.depths[0])
    for p in exp.depths:
        betas, gammas = params.interpolate(p, betas, gammas)


def run_experiment_for_instance(instance: ProblemBase, exp: QAOAExperiment):
    for r in range(exp.repeat):
        for qaoa_run in exp.qaoa:
            qaoa_func = get_qaoa_func(qaoa_run, instance)
            if exp.interpolate:
                run_qaoa_interpolate(instance, qaoa_run, exp)
                continue
            for p in exp.depths:
                pass
    instance.decache()


def run_experiment(exp: QAOAExperiment):
    Problem = None
    for p in ALL_PROBLEMS:
        if p.__name__ == exp.problem.value:
            Problem = p
            break

    instances = Problem.get_instances()
    for size in exp.sizes:
        if exp.instances is not None:
            if size in instances:
                num_instances = len(instances[size])
            else:
                num_instances = 0
            if num_instances < exp.instances:
                raise ValueError(
                    f"Not enough instances: {exp.instances} required, but only {instances} found."
                )
            iter_instances = instances[size][:exp.instances]
        else:
            if size not in instances:
                continue
            iter_instances = instances[size]

        for instance in iter_instances:
            run_experiment_for_instance(instance, exp)



if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise RuntimeError("Expected experiments.toml file")
    config = toml.load(sys.argv[1])

    for k, v in config.items():
        exp = QAOAExperiment(name=k, **v)
        run_experiment(exp)
