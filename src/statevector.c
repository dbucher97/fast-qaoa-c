#include "statevector.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

statevector_t *sv_malloc(uint8_t n_qubits) {
  statevector_t *sv = (statevector_t *)malloc(sizeof(statevector_t));
  sv->data = (cmplx *)malloc((1 << n_qubits) * sizeof(cmplx));
  sv->n_qubits = n_qubits;
  return sv;
}

statevector_t *sv_make_plus_state(uint8_t n_qubits) {
  statevector_t *sv = sv_malloc(n_qubits);

  cmplx val = 1 / sqrt(1 << n_qubits);
  for (size_t i = 0; i < 1 << n_qubits; i++) {
    sv->data[i] = val;
  }

  return sv;
}

statevector_t *sv_clone(const statevector_t *sv) {
  statevector_t *other = sv_malloc(sv->n_qubits);

  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    other->data[i] = sv->data[i];
  }

  return other;
}

void sv_copy(const statevector_t *sv, cmplx *other) {
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    other[i] = sv->data[i];
  }
}

statevector_t *sv_copy_from(const cmplx *other, const uint8_t n_qubits) {
  statevector_t *sv = sv_malloc(n_qubits);
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] = other[i];
  }
  return sv;
}

void sv_free(statevector_t *sv) {
  free(sv->data);
  free(sv);
}

void sv_print(const statevector_t *sv) {
#ifdef USE_FLOAT32
  printf("Printing SV 32\n");
#else
  printf("Printing SV 64\n");
#endif
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    printf("%.3e+%.3ei\n", creal(sv->data[i]), cimag(sv->data[i]));
  }
}

real sv_normalize(statevector_t *sv) {
  real psum = 0.;
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    real re = creal(sv->data[i]);
    real im = cimag(sv->data[i]);
    psum += re * re + im * im;
  }
  real norm = 1 / sqrt(psum);

  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] *= norm;
  }

  return psum;
}

void sv_expec(const statevector_t *sv1, const statevector_t *sv2,
              const diagonals_t *dg, cmplx *res) {
  *res = 0;
  size_t cidx = 0;
  for (size_t i = 0; i < 1 << sv1->n_qubits; i++) {
    cidx = i % (1 << dg->n_qubits);
    *res += conj(sv1->data[i]) * sv2->data[i] * dg->data[i];
  }
}

typedef struct idx_prob_t {
  uint32_t idx;
  real prob;
} idx_prob_t;


int cmp_idx_prob (const void * elem1, const void * elem2){
  const real p1 = ((const idx_prob_t*)elem1)->prob;
  const real p2 = ((const idx_prob_t*)elem2)->prob;
  if(fabs(p1 - p2) < TOL) {
    return 0;
  } else {
    return (p1 > p2) ? 1 : -1;
  }
}

void sv_sample(const statevector_t *sv, const int num, uint32_t *res) {
  idx_prob_t *ip = (idx_prob_t *)malloc(sizeof(idx_prob_t) * (1 << sv->n_qubits));

  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    real re = creal(sv->data[i]);
    real im = cimag(sv->data[i]);
    ip[i].prob = re * re + im * im;
    ip[i].idx = i;
  }

  qsort(ip, 1 << sv->n_qubits, sizeof(idx_prob_t), cmp_idx_prob);

  for (size_t i = 1; i < 1 << sv->n_qubits; i++) {
    ip[i].prob += ip[i - 1].prob;
    // printf("%f\n", ip[i].prob);
  }
  // make sure last value is exactly 1.
  ip[(1 << sv->n_qubits) - 1].prob = 1.f;

  for (int j = 0; j < num; j++) {
    real rval = (double)rand() / (double)RAND_MAX;
    for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
      if (rval < ip[i].prob) {
        res[j] = ip[i].idx;
        break;
      }
    }
  }

  free(ip);
}
