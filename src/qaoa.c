#include "qaoa.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void apply_diagonals(statevector_t *sv, const diagonals_t *dg,
                     const double gamma) {
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] *= cexp(-I * dg->data[i] * gamma);
  }
}

void apply_rx_4radix(statevector_t *sv, const double beta) {
  size_t mask = (1 << sv->n_qubits) - 1;

  double s = sin(beta);
  double c = cos(beta);
  double x = c * c;
  double y = s * s;
  double z = -c * s;

  cmplx *data = sv->data;
  for (size_t q = 0; q < sv->n_qubits; q += 2) {
    for (size_t idx = 0; idx < (1 << (sv->n_qubits - 2)); idx++) {
      size_t i1 = (idx & ~(mask >> q)) << 2 | (idx & (mask >> q));
      size_t i2 = i1 ^ (1 << q);
      size_t i3 = i1 ^ (1 << (q + 1));
      size_t i4 = i3 ^ (1 << q);
      cmplx a = data[i1];
      cmplx b = data[i2];
      cmplx c = data[i3];
      cmplx d = data[i4];
      cmplx bc = cimag(b) + cimag(c) + I * (creal(b) + creal(c));
      cmplx ad = cimag(a) + cimag(d) + I * (creal(a) + creal(d));

      data[i1] = x * a + z * bc + y * d;
      data[i2] = z * ad + x * b + y * c;
      data[i3] = z * ad + x * c + y * b;
      data[i4] = y * a + z * bc + x * d;
    }
  }
}

void apply_rx(statevector_t *sv, const double beta) {
  size_t mask = (1 << sv->n_qubits) - 1;

  cmplx s = -I * sin(beta);
  double c = cos(beta);

  cmplx *data = sv->data;
  for (size_t q = 0; q < sv->n_qubits; q++) {
    for (size_t idx = 0; idx < (1 << (sv->n_qubits - 1)); idx++) {
      size_t i1 = (idx & (mask << q)) << 1 | (idx & ~(mask << q));
      size_t i2 = i1 ^ (1 << q);
      cmplx a = data[i1];
      cmplx b = data[i2];

      data[i1] = c * a + s * b;
      data[i2] = c * b + s * a;
    }
  }
}

void qaoa_inner(statevector_t *sv, const int depth, const diagonals_t *dg,
                const double *betas, const double *gammas) {
  cmplx val = 1 / sqrt(1 << sv->n_qubits);
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] = val;
  }

  for (int p = 0; p < depth; p++) {
    apply_diagonals(sv, dg, gammas[p]);
    apply_rx(sv, betas[p]);
  }
}

statevector_t *qaoa(const int depth, const diagonals_t *dg, const double *betas,
                    const double *gammas) {
  statevector_t *sv = sv_malloc(dg->n_qubits);

  qaoa_inner(sv, depth, dg, betas, gammas);

  return sv;
}

double grad_qaoa_inner(statevector_t *sv_left, statevector_t *sv_right,
                       const int depth, const diagonals_t *dg,
                       const diagonals_t *cost, const double *betas,
                       const double *gammas, double *beta_gradients,
                       double *gamma_gradients) {
  qaoa_inner(sv_left, depth, dg, betas, gammas);
  sv_copy(sv_left, sv_right->data);

  sv_mult(sv_left, cost);

  cmplx buf;
  sv_dot(sv_left, sv_right, &buf);
  double expectation_value = creal(buf);

  for (int p = depth - 1; p >= 0; p--) {
    sum_x_prod(sv_left, sv_right, &beta_gradients[p]);

    apply_rx(sv_left, -betas[p]);
    apply_rx(sv_right, -betas[p]);

    sv_expec(sv_left, sv_right, dg, &buf);
    gamma_gradients[p] = 2 * cimag(buf);

    apply_diagonals(sv_left, dg, -gammas[p]);
    apply_diagonals(sv_right, dg, -gammas[p]);
  }

  return expectation_value;
}

double grad_qaoa(const int depth, const diagonals_t *dg,
                 const diagonals_t *cost, const double *betas,
                 const double *gammas, double *beta_gradients,
                 double *gamma_gradients) {
  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);

  double expectation_value =
      grad_qaoa_inner(sv_left, sv_right, depth, dg, cost, betas, gammas,
                      beta_gradients, gamma_gradients);

  sv_free(sv_left);
  sv_free(sv_right);

  return expectation_value;
}
