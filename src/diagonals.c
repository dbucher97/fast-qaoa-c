#include "diagonals.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define FIND_MIN_VAL(n)                                                        \
  if (n->data[i] < n->min_val)                                                 \
    n->min_val = n->data[i];
#define FIND_MAX_VAL(n)                                                        \
  if (n->data[i] > n->max_val)                                                 \
    n->max_val = n->data[i];

#define FEASIBLE                                                               \
  diff = lhs->data[i] - rhs;                                                   \
  cmp_ok = ((cmp == LTE) & (diff <= 0)) | ((cmp == GTE) & (diff >= 0)) |       \
           ((cmp == LT) & (diff < 0)) | ((cmp == GT) & (diff > 0)) |           \
           ((cmp == EQ) & (fabs(diff) <= TOL)) |                               \
           ((cmp == NEQ) & (fabs(diff) > TOL));

#define NOT_FEASIBLE                                                           \
  diff = lhs->data[i] - rhs;                                                   \
  cmp_ok = ((cmp == LTE) & !(diff <= 0)) | ((cmp == GTE) & !(diff >= 0)) |     \
           ((cmp == LT) & !(diff < 0)) | ((cmp == GT) & !(diff > 0)) |         \
           ((cmp == EQ) & !(fabs(diff) <= TOL)) |                              \
           ((cmp == NEQ) & !(fabs(diff) > TOL));

diagonals_t *dg_malloc(uint8_t n_qubits) {
  diagonals_t *dg = (diagonals_t *)malloc(sizeof(diagonals_t));
  dg->data = (real *)malloc((1 << n_qubits) * sizeof(real));
  dg->n_qubits = n_qubits;
  dg->min_val = INFINITY;
  dg->max_val = -INFINITY;
  return dg;
}

diagonals_t *dg_clone(const diagonals_t *dg) {
  diagonals_t *other = dg_malloc(dg->n_qubits);

  for (size_t i = 0; i < 1 << dg->n_qubits; i++) {
    other->data[i] = dg->data[i];
  }

  other->min_val = dg->min_val;
  other->max_val = dg->max_val;

  return other;
}

void dg_copy(const diagonals_t *dg, real *other) {
  for (size_t i = 0; i < 1 << dg->n_qubits; i++) {
    other[i] = dg->data[i];
  }
}

diagonals_t *dg_copy_from(const real *other, const uint8_t n_qubits) {
  diagonals_t *dg = dg_malloc(n_qubits);

  for (size_t i = 0; i < 1 << dg->n_qubits; i++) {
    dg->data[i] = other[i];
    FIND_MIN_VAL(dg)
    FIND_MAX_VAL(dg)
  }
  return dg;
}

void dg_free(diagonals_t *dg) {
  free(dg->data);
  free(dg);
}

void dg_print(const diagonals_t *dg) {
  for (size_t i = 0; i < 1 << dg->n_qubits; i++) {
    printf("%.3e\n", dg->data[i]);
  }
}

diagonals_t *dg_brute_force(uint8_t n_qubits, int polylen, uint64_t *polykeys,
                            real *polyvals) {
  diagonals_t *dg = dg_malloc(n_qubits);
  for (uint64_t i = 0; i < 1 << n_qubits; i++) {
    real buf = 0.;
    for (int j = 0; j < polylen; j++) {
      buf += ((i & polykeys[j]) == polykeys[j]) * polyvals[j];
    }
    dg->data[i] = buf;
    FIND_MIN_VAL(dg)
    FIND_MAX_VAL(dg)
  }
  return dg;
}

diagonals_t *dg_mask(diagonals_t *dg, diagonals_t *lhs, real rhs, cmp_kind cmp,
                     real val) {
  diagonals_t *ret = dg_malloc(dg->n_qubits);

  bool cmp_ok;
  real diff;
  for (size_t i = 0; i < 1 << dg->n_qubits; i++) {
    NOT_FEASIBLE
    if (cmp_ok) {
      ret->data[i] = val;
    } else {
      ret->data[i] = dg->data[i];
    }
    FIND_MIN_VAL(ret)
    FIND_MAX_VAL(ret)
  }

  return ret;
}

diagonals_t *dg_quad_penalty(diagonals_t *dg, diagonals_t *lhs, real rhs,
                             cmp_kind cmp, real *penalty) {
  diagonals_t *ret = dg_clone(dg);
  real diff;
  bool cmp_ok;

  // auto penalty
  if (*penalty < 0) {
    real scnd_min_val = dg->max_val + 1;
    real min_val = dg->max_val + 1;
    for (size_t i = 0; i < 1 << dg->n_qubits; i++) {
      FEASIBLE
      if (cmp_ok) {
        if (dg->data[i] < min_val) {
          scnd_min_val = min_val;
          min_val = dg->data[i];
        } else if (dg->data[i] > min_val + TOL && dg->data[i] < scnd_min_val) {
          scnd_min_val = dg->data[i];
        }
      }
    }

    if (scnd_min_val > dg->max_val)
      scnd_min_val = 0;

    *penalty = 0.;
    real new_penalty = 0;
    for (size_t i = 0; i < 1 << dg->n_qubits; i++) {
      NOT_FEASIBLE
      if (cmp_ok) {
        new_penalty = (scnd_min_val - dg->data[i]) / (diff * diff + TOL);
        if (new_penalty > *penalty) {
          *penalty = new_penalty;
        }
      }
    }
  }

  ret->min_val = INFINITY;
  ret->max_val = -INFINITY;
  for (size_t i = 0; i < 1 << dg->n_qubits; i++) {
    NOT_FEASIBLE
    if (cmp_ok) {
      ret->data[i] += *penalty * diff * diff;
    }
    FIND_MIN_VAL(ret)
    FIND_MAX_VAL(ret)
  }
  return ret;
}

diagonals_t *dg_cmp(diagonals_t *lhs, real rhs, cmp_kind cmp) {
  diagonals_t *ret = dg_malloc(lhs->n_qubits);
  ret->min_val = 0.;
  ret->max_val = 1.;

  real diff;
  bool cmp_ok;
  for (size_t i = 0; i < 1 << lhs->n_qubits; i++) {
    FEASIBLE
    ret->data[i] = (real)cmp_ok;
  }

  return ret;
}

void dg_scale(diagonals_t *diags, real f) {
  for (size_t i = 0; i < 1 << diags->n_qubits; i++) {
    diags->data[i] *= f;
  }
  if (f < 0) {
    real mv = diags->max_val;
    diags->max_val = f * diags->min_val;
    diags->min_val = f * mv;
  } else {
    diags->max_val *= f;
    diags->min_val *= f;
  }
}

void dg_shift(diagonals_t *diags, real f) {
  for (size_t i = 0; i < 1 << diags->n_qubits; i++) {
    diags->data[i] += f;
  }
  diags->min_val += f;
  diags->max_val += f;
}

real dg_expec_sample(const diagonals_t *diags, const int num,
                     const uint32_t *samples) {
  real res = 0;

  for (int i = 0; i < num; i++) {
    res += diags->data[samples[i]];
  }
  return res / (real)num;
}
