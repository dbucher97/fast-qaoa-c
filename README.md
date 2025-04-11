# FastQAOA

A fast and lightweight Quantum Approximate Optimization Ansatz simulator with fast
differentiation is written in C. Also contains experiments and code for the IF-QAOA
paper and approximate indicator functions.

## Installation

To install, we recommend uv as python package manager. FastQAOA depends on a system-wide
installation of `liblbfgs`. Make sure it is installed via your package manager and `ld`
can link against it. Then
```
make
```
should compile all the C sources.

Afterwards, FastQAOA should be runnable. For instance, run the tests
```
uv run pytest
```
uv should automatically init a new virtual environment and install the dependencies
accordingly.

Note that the C build step is currently not integrated with the uv workflow.

## Concept

FastQAOA relies on the fact that the optimization problem gets brute forced in each
application of the problem Hamiltonian evolution of QAOA. In deep circuits, as well as,
for parameter optimization, the brute-forced result can be stored and directly applied by
multiplication on the state vector.

Furthermore, the application of the mixer Hamiltonian, commonly an application of
Pauli-X-rotations on all qubits have a similar structure to the Fast-Fourier-Transform.
Only O(N log N) operations are therefore required for one iteration. Likewise to FFT,
the simultaneous application of two gates may save some operations (In FFT these are
called radix-4-butterflies). This implementation prefers two-gate applications over
single-gate ones.

## Workflow

### Diagonalization

As the main part of FastQAOA is the stored and diagonalized Hamiltonian, we start by
brute-forcing the optimization problem to hand. In the following, we do that with an
exemplary Max-Cut problem. Max-Cut is not further introduced here, for more
information, see [here](https://en.wikipedia.org/Maximum_cut).

The problem is defined on a graph
```python
import networkx as nx

G = nx.erdos_renyi_graph(6, 0.5, seed=8001)

# build the terms of the hamiltonian
terms = {e: 2. for e in G.edges()}
terms.update({(v,): -float(G.degree(v)) for v in G.nodes()})

# terms = {
#   (0, 4): 2.0, (1, 3): 2.0, (1, 4): 2.0, (1, 5): 2.0, (2, 4): 2.0, (2, 5): 2.0,
#   (3, 5): 2.0, (0,): -1.0, (1,): -3.0, (2,): -2.0, (3,): -2.0, (4,): -3.0, (5,): -1.0
# }
```

The Hamiltonian is the sum of the product of binary variables times bias. The product is
given as tuple keys of a dictionary. Now, utilizing the `Diagonals` object, we can brute
force the problem, as follows

```python
from fastqaoa import Diagonals

dg = Diagonals.brute_force_hamiltonian(6, terms)

# dg.to_numpy() -> [0. -1. -3. ... -3. -1.  0.]
```

### QAOA application

The QAOA circuit is paramterized by `p` betas and gammas paramters, for now hardcoded
as. Yet, `fastqoaa.params` features parameter initialization methods. The full QAOA
circuit can be computed by simply
```python
from fastqaoa import qaoa

betas = [0.9, 0.5, 0.1]
gammas = [0.1, 0.5, 0.9]

sv = qaoa(dg, betas, gammas)
```
where `sv` is a `Statevector` object. We can either compute the expectation value form
it by
```python
print(dg.expec(sv))
# -4.922789608617443
```
or sample form the state vector probabilities, i.e. measuring the prepared state
frequently
```python
samples = sv.sample(100)

# The expectation value can also be computed from the samples
print(dg.expec(samples))
# -4.95
```

### Parameter Optimization

The parameters can be optimized using the cost function from beforehand and
gradient-free optimizers. Alternatively, we can use gradient-based optimizers. In a QC
application, that would require gradient computation through methods like parameter
shift. Nevertheless, in simulation, we can rely on computing the exact gradient
```python
from fastqaoa import grad_qaoa

val, grad_betas, grad_gammas = grad_qaoa(dg, dg, betas, gammas)
# val = -4.922789608617443
# grad_betas = [-0.11524348  1.19118023 -3.51788873]
# grad_gammas = [ 0.75853451 -0.8354133  -0.11325605]
```
The second `dg` in `grad_qaoa` refers to the Hamiltonian, w.r.t. wich the derived
expectation values is calculated. In principle, this one can deviate from the Problem
Hamiltonian.

FastQAOA features a built-in parameter optimizer based on L-BFGS. The limited memory
part is not really important, as the number of parameters considered is generally
not the bottleneck. However, `liblbfgs` is an excellent implementation of the BFGS procedure.

To optimize parameters, simply write
```python
from fastqaoa import optimize_qaoa_lbfgs

res = optimize_qaoa_lbfgs(dg, dg, betas, gammas)

# LBFGSResult(status=<LBFGSStatus.Success: 0>, it=29, betas=[1.43203, 0.43962492, 0.2790024],
# gammas=[-0.83682539,  1.5541561 ,  1.02302044], calls=39, log=[-5.27183532, -5.3352494 , ..., -5.60470587])
```

### Evaluate results

FastQAOA comes with a metrics tool that automatically computes most of the important
metrics for QAOA results. It is used as shown below

```python
from fastqaoa import Metrics

sv = qaoa(sv, res.betas, res.gammas)

constr = Diagonals.array(np.ones(1 << 6)) # if states are infeasible mark with 0 inestead of 1
metrics = Metrics.compute(sv, dg, constr)


print(metrics.dump())
# {'energy': -5.6047058, 'approx_ratio': 0.9341176, 'p_opt': 0.713482, ...}
```


### IF-QAOA

TODO

## Benchmarks

FastQAOA is currently only focused on single-core performance (multi-core is yet to come
but no focus). The benchmarks are run against
[QOKit](https://github.com/jpmorganchase/QOKit),
[Qiskit](https://www.ibm.com/quantum/qiskit) (Aer Statevector simulator) and
[Pennylane](https://pennylane.ai/) (Lightning CPU). All simulators are set to use
single-core only. The Benchmarks are run on an Apple M1 Mac, FastQAOA and QOKit were
compiled with Apple Clang with optimization level O3.

First, let's compare simulation time (left: including brute forcing, right: without brute
force) at QAOA depth 6:

<img src="assets/p6_comp_tot_time.png" width="49%" /><img src="assets/p6_comp.png" width="49%"  />

As apparent, FastQAOA performs the fastest out of all. However, in pure simulation time, QOKit is on par with FastQAOA.
The full circuit simulators are considerably worse in comparison. Let's look at simulation time wrt. layer depth at size 24:

<img src="assets/n24_comp.png" width="50%" />

As expected, the simulation time grows linearly, with QOKit and FastQAOA exhibiting similar performance.


### A look at gradients

Unlike QOKit, FastQAOA allows for gradient computation based on the adjoint method. 
Pennylane's Lightning simulator also estimates gradients through the same method.
Therefore, we here benchmark gradients of the two methods

<img src="assets/p6_gradient_comp.png" width="49%" /><img src="assets/n18_grad_comp.png" width="49%"  />

FastQAOA is orders of magnitude faster than Pennylane, allowing for fast, exact, optimization of QAOA parameters.

## Results of IF-QAOA Paper

The code, data and notebooks to recreate the plots in the IF-QAOA paper are all provided
within this repo. To run an experiment, use the `experiment_runner` provided via
`problems`. Each experiment is specified in a `toml` file.
```
uv run problems/experiment_runner.py experiments/int_main.toml
```

All the results are stored in the `results` directory. The problem instances themselves
are stored in `data`. To access problem instances, e.g. integer-valued Knapsack
instances

```python
from problems import IntegerKnapsack

instances: dict = IntegerKnapsack.load_instances()

N = 12   # problem size
idx = 0  # instance index

instance: IntegerKnapsack = instances[N][idx]

# get brute-forced f, g
f, g = instance.diagonalized()

# get brute-forced IF cost \tilde{f}
tilde_f = instance.masked_cost()

# get brute-forced quadratic penalty
f2 = instance.quad_penalty()

# get brute-forced quadratic penalty with slack variables
f3 = instance.quad_penalty_full_problem()
```

To reproduce the results of the paper. Simply execute the notebooks in the `notebooks`
dir by running:
```
uv run jupyter lab
```


