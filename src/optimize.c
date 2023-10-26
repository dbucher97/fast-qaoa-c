#include "diagonals.h"
#include "qaoa.h"
#include "statevector.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPS 1e-8
#define BETA1 0.9
#define BETA2 0.99

#define NEW_ADAM_STATE                                                         \
  { NULL, NULL, NULL, NULL, 0, NULL, lr, EPS, tol, BETA1, BETA2, maxiter }

typedef struct adam_state_t {
  double *m_betas;
  double *m_gammas;
  double *w_betas;
  double *w_gammas;
  int it;
  double *trace;
  const double lr;
  const double eps;
  const double tol;
  const double beta1;
  const double beta2;
  const int maxiter;
} adam_state_t;

void opt_init_adam_state(adam_state_t *state, const double lr,
                         const int maxiter, const double tol, const int depth) {
  state->m_betas = (double *)malloc(sizeof(double) * depth);
  state->m_gammas = (double *)malloc(sizeof(double) * depth);
  state->w_betas = (double *)malloc(sizeof(double) * depth);
  state->w_gammas = (double *)malloc(sizeof(double) * depth);
  state->trace = (double *)malloc(sizeof(double) * maxiter);

  for (int i = 0; i < depth; i++) {
    state->m_betas[i] = 0;
    state->m_gammas[i] = 0;
    state->w_betas[i] = 0;
    state->w_gammas[i] = 0;
  }
}

void opt_free_adam_state(adam_state_t *state) {
  free(state->m_betas);
  free(state->m_gammas);
  free(state->w_betas);
  free(state->w_gammas);
  free(state->trace);
}

void opt_optimize_qaoa(const int depth, const diagonals_t *dg,
                       const diagonals_t *cost, double *betas, double *gammas,
                       double lr, int maxiter, double tol) {
  double *grad_betas = (double *)malloc(sizeof(double) * depth);
  double *grad_gammas = (double *)malloc(sizeof(double) * depth);

  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);

  adam_state_t state = NEW_ADAM_STATE;
  opt_init_adam_state(&state, lr, maxiter, tol, depth);

  double last_val = INFINITY;

  for (; state.it < state.maxiter;) {
    state.trace[state.it] =
        grad_qaoa_inner(sv_left, sv_right, depth, dg, cost, betas, gammas,
                        grad_betas, grad_gammas);

    if (fabs(last_val - state.trace[state.it]) < tol)
      break;
    last_val = state.trace[state.it];

    state.it += 1;

    double lrit = state.lr * sqrt(1 - pow(state.beta2, state.it)) /
                  (1 - pow(state.beta1, state.it));

    for (int p = 0; p < depth; p++) {
      state.m_betas[p] =
          state.beta1 * state.m_betas[p] + (1 - state.beta1) * grad_betas[p];
      state.w_betas[p] = state.beta2 * state.w_betas[p] +
                         (1 - state.beta2) * grad_betas[p] * grad_betas[p];
      betas[p] -= lrit * state.m_betas[p] / (sqrt(state.w_betas[p]) + state.eps);

      state.m_gammas[p] =
          state.beta1 * state.m_gammas[p] + (1 - state.beta1) * grad_gammas[p];
      state.w_gammas[p] = state.beta2 * state.w_gammas[p] +
                          (1 - state.beta2) * grad_gammas[p] * grad_gammas[p];
      gammas[p] -= lrit * state.m_gammas[p] / (sqrt(state.w_gammas[p]) + state.eps);
    }

  }

  opt_free_adam_state(&state);
  free(grad_betas);
  free(grad_gammas);

  sv_free(sv_left);
  sv_free(sv_right);
}
