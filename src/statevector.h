#include "mtypes.h"
#include "diagonals.h"
#include <stdint.h>
#include <stdlib.h>

#ifndef _STATEVECTOR
#define _STATEVECTOR

typedef struct statevector_t {
  uint8_t n_qubits;
  cmplx *data;
} statevector_t;

statevector_t *sv_malloc(uint8_t n_qubits);

statevector_t *sv_make_plus_state(uint8_t n_qubits);

statevector_t *sv_clone(const statevector_t *sv);

void sv_copy(const statevector_t *sv, cmplx *other);

statevector_t* sv_copy_from(const cmplx *other, const uint8_t n_qubits);

void sv_free(statevector_t *sv);

void sv_print(const statevector_t *sv);

real sv_normalize(statevector_t *sv);

inline void sv_mult(statevector_t *sv, const diagonals_t *dg) {
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] *= dg->data[i];
  }
}

inline void sv_dot(const statevector_t *sv1, const statevector_t *sv2, cmplx* res) {
  *res = 0;
  for (size_t i = 0; i < 1 << sv1->n_qubits; i++) {
    *res += conj(sv1->data[i]) * sv2->data[i];
  }
}

inline void sv_expec(const statevector_t *sv1, const statevector_t *sv2,
                      const diagonals_t *dg, cmplx* res) {
  *res = 0;
  for (size_t i = 0; i < 1 << sv1->n_qubits; i++) {
    *res += conj(sv1->data[i]) * sv2->data[i] * dg->data[i];
  }
}

#endif
