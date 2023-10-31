#include <lbfgs.h>
#include <stdio.h>

#include "diagonals.h"
#include "frx.h"
#include "qaoa.h"
#include "qpe_qaoa.h"
#include "statevector.h"

typedef struct {
  statevector_t *sv_left;
  statevector_t *sv_right;
  statevector_t *sv_left_p;
  frx_plan_t *plan;
  const diagonals_t *dg;
  const diagonals_t *cost;
  const diagonals_t *constr;
} qaoa_instance_t;

static lbfgsfloatval_t evaluate_qaoa(void *instance, const lbfgsfloatval_t *x,
                                     lbfgsfloatval_t *g, const int n,
                                     const lbfgsfloatval_t step) {
  lbfgsfloatval_t expectation_value;

  qaoa_instance_t *qaoa_instance = (qaoa_instance_t *)instance;

  const int p = n >> 1;

  grad_qaoa_inner(qaoa_instance->sv_left, qaoa_instance->sv_right,
                  qaoa_instance->plan, p, qaoa_instance->dg,
                  qaoa_instance->cost, x, &x[p], g, &g[p], &expectation_value);

  return expectation_value;
}

static lbfgsfloatval_t evaluate_qpe_qaoa(void *instance,
                                         const lbfgsfloatval_t *x,
                                         lbfgsfloatval_t *g, const int n,
                                         const lbfgsfloatval_t step) {
  lbfgsfloatval_t expectation_value;

  double psucc;

  qaoa_instance_t *qaoa_instance = (qaoa_instance_t *)instance;

  const int p = n >> 1;

  grad_qpe_qaoa_inner(
      qaoa_instance->sv_left, qaoa_instance->sv_right, qaoa_instance->sv_left_p,
      qaoa_instance->plan, p, qaoa_instance->dg, qaoa_instance->cost,
      qaoa_instance->constr, x, &x[p], g, &g[p], &psucc, &expectation_value);

  return expectation_value;
}

// static int progress(void *instance, const lbfgsfloatval_t *x,
//                     const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
//                     const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
//                     const lbfgsfloatval_t step, int n, int k, int ls) {
//   printf("Iteration %d:\n", k);
//   printf("  fx = %f, xnorm = %f, gnorm = %f, step = %f\n", fx, xnorm, gnorm,
//          step);
//   printf("\n");
//   return 0;
// }

int opt_lbfgs_qaoa(int depth, const diagonals_t *dg, const diagonals_t *cost,
                   double *betas, double *gammas, int max_iter) {
  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);
  frx_plan_t* plan = frx_make_plan(sv_left, RDX4);

  qaoa_instance_t instance = {sv_left, sv_right, NULL, plan, dg, cost, NULL};

  const int N = depth << 1;

  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(N);
  lbfgs_parameter_t param;
  if (x == NULL) {
    printf("ERROR: Failed to allocate a memory block for variables.\n");
    return -1;
  }
  for (int i = 0; i < depth; i++) {
    x[i] = betas[i];
    x[depth + i] = gammas[i];
  }
  lbfgs_parameter_init(&param);
  param.max_iterations = max_iter;

  int ret;
  ret = lbfgs(N, x, &fx, evaluate_qaoa, NULL, &instance, &param);

  for (int i = 0; i < depth; i++) {
    betas[i] = x[i];
    gammas[i] = x[depth + i];
  }

  lbfgs_free(x);
  sv_free(sv_left);
  sv_free(sv_right);
  frx_free(plan);
  return ret;
}

int opt_lbfgs_qpe_qaoa(int depth, const diagonals_t *dg,
                       const diagonals_t *cost, const diagonals_t *constr,
                       double *betas, double *gammas, int max_iter) {
  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);
  statevector_t *sv_left_p = sv_malloc(dg->n_qubits);
  frx_plan_t* plan = frx_make_plan(sv_left, RDX4);

  qaoa_instance_t instance = {sv_left, sv_right, sv_left_p, plan, dg, cost, constr};

  const int N = depth << 1;

  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(N);
  lbfgs_parameter_t param;
  if (x == NULL) {
    printf("ERROR: Failed to allocate a memory block for variables.\n");
    return -1;
  }
  for (int i = 0; i < depth; i++) {
    x[i] = betas[i];
    x[depth + i] = gammas[i];
  }
  lbfgs_parameter_init(&param);
  param.max_iterations = max_iter;

  int ret;
  ret = lbfgs(N, x, &fx, evaluate_qpe_qaoa, NULL, &instance, &param);

  for (int i = 0; i < depth; i++) {
    betas[i] = x[i];
    gammas[i] = x[depth + i];
  }

  lbfgs_free(x);
  sv_free(sv_left);
  sv_free(sv_right);
  sv_free(sv_left_p);
  frx_free(plan);
  return ret;
}
