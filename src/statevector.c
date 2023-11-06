#include "statevector.h"
#include <math.h>
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

statevector_t* sv_copy_from(const cmplx *other, const uint8_t n_qubits) {
  statevector_t * sv = sv_malloc(n_qubits);
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

