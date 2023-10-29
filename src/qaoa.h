#include "diagonals.h"
#include "statevector.h"

#ifndef _QAOA
#define _QAOA

inline void sum_x_prod(const statevector_t *sv_left,
                       const statevector_t *sv_right, double *res) {
  *res = 0;
  for (size_t i = 0; i < 1 << sv_left->n_qubits; i++) {
    cmplx s = 0;
    for (uint8_t j = 0; j < sv_right->n_qubits; j++) {
      s += sv_right->data[i ^ (1 << j)];
    }
    *res -= 2 * cimag(conj(s) * sv_left->data[i]);
  }
}

void apply_diagonals(statevector_t *sv, const diagonals_t *dg,
                     const double gamma);
void apply_rx(statevector_t *sv, const double beta);

void qaoa_inner(statevector_t *sv, const int depth, const diagonals_t *dg,
                const double *betas, const double *gammas);

statevector_t *qaoa(const int depth, const diagonals_t *dg, const double *betas,
                    const double *gammas);

double grad_qaoa_inner(statevector_t *sv_left, statevector_t *sv_right, const int depth,
                 const diagonals_t *dg, const diagonals_t *cost,
                 const double *betas, const double *gammas,
                 double *beta_gradients, double *gamma_gradients);

double grad_qaoa(const int depth, const diagonals_t *dg,
                 const diagonals_t *cost, const double *betas,
                 const double *gammas, double *beta_gradients,
                 double *gamma_gradients);

#endif
