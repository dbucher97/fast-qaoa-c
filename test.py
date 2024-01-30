import networkx as nx
import numpy as np

G = nx.erdos_renyi_graph(6, 0.5, seed=8001)

terms = {e: 2. for e in G.edges()}
terms.update({(v,): -float(G.degree(v)) for v in G.nodes()})

###################

from fastqaoa import Diagonals

dg = Diagonals.brute_force_hamiltonian(6, terms)
print(dg.to_numpy())

from fastqaoa import qaoa

betas = [0.9, 0.5, 0.1]
gammas = [0.1, 0.5, 0.9]

sv = qaoa(dg, betas, gammas)

print(dg.expec(sv))

samples = sv.sample(100)

# The expectation value can also be computed form the samples
print(dg.expec(samples)) 


from fastqaoa import grad_qaoa

val, grad_betas, grad_gammas = grad_qaoa(dg, dg, betas, gammas)
print(val, grad_betas, grad_gammas)


from fastqaoa import optimize_qaoa_lbfgs

res = optimize_qaoa_lbfgs(dg, dg, betas, gammas)

print(res)


from fastqaoa import Metrics

sv = qaoa(dg, res.betas, res.gammas)
print(dg.expec(sv))
metrics = Metrics.compute(sv, dg, Diagonals.from_numpy(np.ones(1 << 6)))

print(metrics.dump())
