#include "qaoa.h"
#include "qpe_qaoa.h"
#include <math.h>

#define TOL 1e-8

typedef struct metrics_t {
  real energy;
  real approx_ratio;
  real feas_ratio;
  real feas_approx_ratio;
  real p_opt;
  real p_999;
  real p_99;
  real p_9;
  real rnd_approx_ratio;
  real min_val;
  real rnd_val;
  real max_val;
} metrics_t;

metrics_t *mtr_compute(const statevector_t *sv, const diagonals_t *cost,
                       const diagonals_t *constr) {
  metrics_t *metrics = (metrics_t *)malloc(sizeof(metrics_t));
  real p, c;
  metrics->approx_ratio = 0;
  metrics->feas_ratio = 0;
  metrics->feas_approx_ratio = 0;
  metrics->p_opt = 0;
  metrics->p_999 = 0;
  metrics->p_99 = 0;
  metrics->p_9 = 0;
  metrics->rnd_val = 0;
  size_t icost = 0;
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    p = creal(sv->data[i]) * creal(sv->data[i]) +
        cimag(sv->data[i]) * cimag(sv->data[i]);
    icost = i % (1 << cost->n_qubits);
    c = cost->data[icost] / cost->min_val;
    metrics->energy += p * cost->data[icost];
    metrics->approx_ratio += p * c;

    if (constr->data[icost] >= 0) {
      metrics->feas_ratio += p;
      metrics->feas_approx_ratio += p * c;
    }
    if (1 - c < TOL) {
      metrics->p_opt += p;
    }
    if (1 - c < 1e-3) {
      metrics->p_999 += p;
    }
    if (1 - c < 1e-2) {
      metrics->p_99 += p;
    }
    if (1 - c < 1e-1) {
      metrics->p_9 += p;
    }
    metrics->rnd_val += cost->data[icost];
  }

  metrics->feas_approx_ratio /= metrics->feas_ratio;
  metrics->min_val = cost->min_val;
  metrics->max_val = cost->max_val;
  metrics->rnd_val /= (real)(1 << sv->n_qubits);
  metrics->rnd_approx_ratio = (metrics->rnd_val - metrics->energy) /
                              (metrics->rnd_val - metrics->min_val);
  return metrics;
}

void mtr_free(metrics_t *mtr) { free(mtr); }
