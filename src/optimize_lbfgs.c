#include <stdio.h>

#include "diagonals.h"
#include "frx.h"
#include "mtypes.h"
#include "qaoa.h"
#include "qpe_qaoa.h"
#include "statevector.h"

#include <lbfgs.h>

typedef struct {
  int it;
  int calls;
  statevector_t *sv_left;
  statevector_t *sv_right;
  statevector_t *sv_left_p;
  frx_plan_t *plan;
  const diagonals_t *dg;
  const diagonals_t *cost;
  const diagonals_t *constr;
  real *log;
#if USE_FLOAT32
  float *x;
  float *g;
#endif
} qaoa_instance_t;

static lbfgsfloatval_t evaluate_qaoa(void *instance, const lbfgsfloatval_t *x,
                                     lbfgsfloatval_t *g, const int n,
                                     const lbfgsfloatval_t step) {
  real expectation_value;

  qaoa_instance_t *qaoa_instance = (qaoa_instance_t *)instance;

  const int p = n >> 1;

#if USE_FLOAT32
  for (int i = 0; i < n; i++) {
    qaoa_instance->x[i] = (float)x[i];
  }
#endif

  grad_qaoa_inner(qaoa_instance->sv_left, qaoa_instance->sv_right,
                  qaoa_instance->plan, p, qaoa_instance->dg,
                  qaoa_instance->cost,
#ifdef USE_FLOAT32
                  qaoa_instance->x, qaoa_instance->x + p, qaoa_instance->g,
                  qaoa_instance->g + p,
#else
                  x, x + p, g, g + p,
#endif
                  &expectation_value);

#if USE_FLOAT32
  for (int i = 0; i < n; i++) {
    g[i] = (lbfgsfloatval_t)qaoa_instance->g[i];
  }
#endif

  qaoa_instance->calls++;
  return (lbfgsfloatval_t)expectation_value;
}

static lbfgsfloatval_t evaluate_qpe_qaoa(void *instance,
                                         const lbfgsfloatval_t *x,
                                         lbfgsfloatval_t *g, const int n,
                                         const lbfgsfloatval_t step) {
  real expectation_value;

  real psucc;

  qaoa_instance_t *qaoa_instance = (qaoa_instance_t *)instance;

  const int p = n >> 1;

#if USE_FLOAT32
  for (int i = 0; i < n; i++) {
    qaoa_instance->x[i] = (float)x[i];
  }
#endif

  grad_qpe_qaoa_inner(qaoa_instance->sv_left, qaoa_instance->sv_right,
                      qaoa_instance->sv_left_p, qaoa_instance->plan, p,
                      qaoa_instance->dg, qaoa_instance->cost,
                      qaoa_instance->constr,
#ifdef USE_FLOAT32
                      qaoa_instance->x, qaoa_instance->x + p, qaoa_instance->g,
                      qaoa_instance->g + p,
#else
                      x, x + p, g, g + p,
#endif
                      &psucc, &expectation_value);

#if USE_FLOAT32
  for (int i = 0; i < n; i++) {
    g[i] = (lbfgsfloatval_t)qaoa_instance->g[i];
  }
#endif

  qaoa_instance->calls++;
  return (lbfgsfloatval_t)expectation_value;
}

static int progress(void *instance, const lbfgsfloatval_t *x,
                    const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                    const lbfgsfloatval_t step, int n, int k, int ls) {
  qaoa_instance_t *qaoa = instance;
  qaoa->it = k;
  qaoa->log[k - 1] = (real)fx;
  // printf("Iteration %d:\n", k);
  // printf("  fx = %f, xnorm = %f, gnorm = %f, step = %f\n", fx, xnorm, gnorm,
  //        step);
  // printf("\n");
  return 0;
}

void opt_parameter_init(lbfgs_parameter_t *param, const int *max_iter,
                        const double *tol, const int *linesearch,
                        const int *m) {
  lbfgs_parameter_init(param);
  if (max_iter != NULL)
    param->max_iterations = *max_iter;
  if (linesearch != NULL)
    param->linesearch = *linesearch;
  if (tol != NULL) {
    param->epsilon = *tol;
    param->ftol = *tol;
  }
  if (m != NULL)
    param->m = *m;
}

int opt_lbfgs_qaoa(int depth, const diagonals_t *dg, const diagonals_t *cost,
                   real *betas, real *gammas, int *it, int *calls, real *log,
                   const int *max_iter, const double *tol,
                   const int *linesearch, const int *m) {
  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);
  frx_plan_t *plan = frx_make_plan(sv_left, RDX4);

  qaoa_instance_t instance = {
      0, 0, sv_left, sv_right, NULL, plan, dg, cost, NULL, log,
  };
#ifdef USE_FLOAT32
  instance.x = malloc(2 * depth * sizeof(float));
  instance.g = malloc(2 * depth * sizeof(float));
#endif

  const int N = depth << 1;

  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(N);
  if (x == NULL) {
    printf("ERROR: Failed to allocate a memory block for variables.\n");
    return -1;
  }
  for (int i = 0; i < depth; i++) {
    x[i] = betas[i];
    x[depth + i] = gammas[i];
  }
  lbfgs_parameter_t param;
  opt_parameter_init(&param, max_iter, tol, linesearch, m);

  int ret;
  ret = lbfgs(N, x, &fx, evaluate_qaoa, progress, &instance, &param);

  for (int i = 0; i < depth; i++) {
    betas[i] = x[i];
    gammas[i] = x[depth + i];
  }
  *it = instance.it;
  *calls = instance.calls;

  lbfgs_free(x);
  sv_free(sv_left);
  sv_free(sv_right);
  frx_free(plan);
#ifdef USE_FLOAT32
  free(instance.x);
  free(instance.g);
#endif
  return ret;
}

int opt_lbfgs_qpe_qaoa(int depth, const diagonals_t *dg,
                       const diagonals_t *cost, const diagonals_t *constr,
                       real *betas, real *gammas, int *it, int *calls,
                       real *log, const int *max_iter, const double *tol,
                       const int *linesearch, const int *m) {
  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);
  statevector_t *sv_left_p = sv_malloc(dg->n_qubits);
  frx_plan_t *plan = frx_make_plan(sv_left, RDX4);

  qaoa_instance_t instance = {0,    0,  sv_left, sv_right, sv_left_p,
                              plan, dg, cost,    constr,   log};
#ifdef USE_FLOAT32
  instance.x = malloc(2 * depth * sizeof(float));
  instance.g = malloc(2 * depth * sizeof(float));
#endif

  const int N = depth << 1;

  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(N);
  if (x == NULL) {
    printf("ERROR: Failed to allocate a memory block for variables.\n");
    return -1;
  }
  for (int i = 0; i < depth; i++) {
    x[i] = betas[i];
    x[depth + i] = gammas[i];
  }
  lbfgs_parameter_t param;
  opt_parameter_init(&param, max_iter, tol, linesearch, m);

  int ret;
  ret = lbfgs(N, x, &fx, evaluate_qpe_qaoa, progress, &instance, &param);

  for (int i = 0; i < depth; i++) {
    betas[i] = x[i];
    gammas[i] = x[depth + i];
  }
  *it = instance.it;
  *calls = instance.calls;

  lbfgs_free(x);
  sv_free(sv_left);
  sv_free(sv_right);
  sv_free(sv_left_p);
  frx_free(plan);
#ifdef USE_FLOAT32
  free(instance.x);
  free(instance.g);
#endif
  return ret;
}
