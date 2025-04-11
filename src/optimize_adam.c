#include "diagonals.h"
#include "frx.h"
#include "mtypes.h"
#include "qaoa.h"
#include "qpe_qaoa.h"
#include "statevector.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPS 1e-8
#define BETA1 0.9
#define BETA2 0.999

typedef enum {
  ADAM_CONVERGED = 0,
  ADAM_MAX_IT,
} adam_result_t;

#define NEW_ADAM_STATE                                                         \
  { NULL, NULL, NULL, NULL, 0, NULL, lr, EPS, tol, BETA1, BETA2, maxiter }

typedef struct adam_state_t {
  real *m_betas;
  real *m_gammas;
  real *w_betas;
  real *w_gammas;
  int it;
  real *trace;
  const real lr;
  const real eps;
  const real tol;
  const real beta1;
  const real beta2;
  const int maxiter;
} adam_state_t;

void opt_init_adam_state(adam_state_t *state, const real lr, const int maxiter,
                         const real tol, const int depth) {
  state->m_betas = (real *)malloc(sizeof(real) * depth);
  state->m_gammas = (real *)malloc(sizeof(real) * depth);
  state->w_betas = (real *)malloc(sizeof(real) * depth);
  state->w_gammas = (real *)malloc(sizeof(real) * depth);
  state->trace = (real *)malloc(sizeof(real) * maxiter);

  for (int i = 0; i < depth; i++) {
    state->m_betas[i] = 0.;
    state->m_gammas[i] = 0.;
    state->w_betas[i] = 0.;
    state->w_gammas[i] = 0.;
  }
}

void opt_free_adam_state(adam_state_t *state) {
  free(state->m_betas);
  free(state->m_gammas);
  free(state->w_betas);
  free(state->w_gammas);
  free(state->trace);
}

adam_result_t opt_adam_qaoa(const int depth, const diagonals_t *dg,
                            const diagonals_t *cost, real *betas, real *gammas,
                            real lr, const int maxiter, const real tol,
                            int *it) {
  // printf("=== OPTIM ================================================\n");
  // printf("betas: %f", betas[0]);
  // for (int p = 1; p < depth; p++) {
  //   printf(", %f", betas[p]);
  // }
  // printf("\n");
  // printf("gammas: %f", gammas[0]);
  // for (int p = 1; p < depth; p++) {
  //   printf(", %f", gammas[p]);
  // }
  // printf("\n");

  real *grad_betas = (real *)malloc(sizeof(real) * depth);
  real *grad_gammas = (real *)malloc(sizeof(real) * depth);

  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);

  frx_plan_t *plan = frx_make_plan(sv_left, RDX4);

  adam_state_t state = NEW_ADAM_STATE;
  opt_init_adam_state(&state, lr, maxiter, tol, depth);

  real last_val = INFINITY;

  for (; state.it < state.maxiter;) {
    grad_qaoa_inner(sv_left, sv_right, plan, depth, dg, cost, betas, gammas,
                    grad_betas, grad_gammas, &state.trace[state.it]);

    state.it += 1;

    real lrit = state.lr * sqrt(1 - pow(state.beta2, state.it)) /
                (1 - pow(state.beta1, state.it));

    // printf("grad beta: %f", grad_betas[0]);
    // for (int p = 1; p < depth; p++) {
    //   printf(", %f", grad_betas[p]);
    // }
    // printf("\n");
    // printf("grad gamma: %f", grad_gammas[0]);
    // for (int p = 1; p < depth; p++) {
    //   printf(", %f", grad_gammas[p]);
    // }
    // printf("\n");

    for (int p = 0; p < depth; p++) {
      state.m_betas[p] =
          state.beta1 * state.m_betas[p] + (1 - state.beta1) * grad_betas[p];
      state.w_betas[p] = state.beta2 * state.w_betas[p] +
                         (1 - state.beta2) * grad_betas[p] * grad_betas[p];
      // printf("%f, ",
      //        lrit * state.m_betas[p] / (sqrt(state.w_betas[p]) + state.eps));
      betas[p] -=
          lrit * state.m_betas[p] / (sqrt(state.w_betas[p]) + state.eps);

      state.m_gammas[p] =
          state.beta1 * state.m_gammas[p] + (1 - state.beta1) * grad_gammas[p];
      state.w_gammas[p] = state.beta2 * state.w_gammas[p] +
                          (1 - state.beta2) * grad_gammas[p] * grad_gammas[p];
      // printf("%f, ",
      //        lrit * state.m_gammas[p] / (sqrt(state.w_gammas[p]) + state.eps));
      // printf("%f, %f\n", lrit, lr);
      gammas[p] -=
          lrit * state.m_gammas[p] / (sqrt(state.w_gammas[p]) + state.eps);
    }

    // printf("%d: %f\n", state.it - 1,
    //        fabs(last_val - state.trace[state.it - 1]));
    if (fabs(last_val - state.trace[state.it - 1]) <= tol)
      break;
    last_val = state.trace[state.it - 1];
  }
  *it = state.it;
  adam_result_t ret;
  if (state.it == state.maxiter) {
    ret = ADAM_MAX_IT;
  } else {
    ret = ADAM_CONVERGED;
  }

  opt_free_adam_state(&state);
  free(grad_betas);
  free(grad_gammas);

  sv_free(sv_left);
  sv_free(sv_right);
  frx_free(plan);
  return ret;
}

adam_result_t opt_adam_qpe_qaoa(const int depth, const diagonals_t *dg,
                                const diagonals_t *cost,
                                const diagonals_t *constr, real *betas,
                                real *gammas, real lr, const int maxiter,
                                const real tol, int *it) {
  real *grad_betas = (real *)malloc(sizeof(real) * depth);
  real *grad_gammas = (real *)malloc(sizeof(real) * depth);

  // printf("ALSO CHANGE OPT QPE QAOA!\n");
  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);
  statevector_t *sv_left_p = sv_malloc(dg->n_qubits);

  frx_plan_t *plan = frx_make_plan(sv_left, RDX4);

  adam_state_t state = NEW_ADAM_STATE;
  opt_init_adam_state(&state, lr, maxiter, tol, depth);

  real last_val = INFINITY;

  real psucc;

  for (; state.it < state.maxiter;) {
    grad_qpe_qaoa_inner(sv_left, sv_right, sv_left_p, plan, depth, dg, cost,
                        constr, betas, gammas, grad_betas, grad_gammas, &psucc,
                        &state.trace[state.it]);

    if (fabs(last_val - state.trace[state.it]) < tol)
      break;
    last_val = state.trace[state.it];

    state.it += 1;

    real lrit = state.lr * sqrt(1 - pow(state.beta2, state.it)) /
                (1 - pow(state.beta1, state.it));

    for (int p = 0; p < depth; p++) {
      state.m_betas[p] =
          state.beta1 * state.m_betas[p] + (1 - state.beta1) * grad_betas[p];
      state.w_betas[p] = state.beta2 * state.w_betas[p] +
                         (1 - state.beta2) * grad_betas[p] * grad_betas[p];
      betas[p] -=
          lrit * state.m_betas[p] / (sqrt(state.w_betas[p]) + state.eps);

      state.m_gammas[p] =
          state.beta1 * state.m_gammas[p] + (1 - state.beta1) * grad_gammas[p];
      state.w_gammas[p] = state.beta2 * state.w_gammas[p] +
                          (1 - state.beta2) * grad_gammas[p] * grad_gammas[p];
      gammas[p] -=
          lrit * state.m_gammas[p] / (sqrt(state.w_gammas[p]) + state.eps);
    }
  }
  *it = state.it;
  adam_result_t ret;
  if (state.it == state.maxiter) {
    ret = ADAM_MAX_IT;
  } else {
    ret = ADAM_CONVERGED;
  }

  opt_free_adam_state(&state);
  free(grad_betas);
  free(grad_gammas);

  sv_free(sv_left);
  sv_free(sv_right);
  sv_free(sv_left_p);
  frx_free(plan);
  return ret;
}
