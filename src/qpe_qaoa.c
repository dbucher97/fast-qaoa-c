#include "qpe_qaoa.h"
#include "qaoa.h"
#include <math.h>
#include <stdio.h>

#define EPS 1e-6

void apply_qpe_diagonals_normalized(statevector_t *sv, const diagonals_t *dg,
                                    const diagonals_t *constr,
                                    const double gamma, double *psum) {
  cmplx buf;
  double norm;
  *psum = 0;
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    buf = 0.5 * cexp(-I * dg->data[i] * gamma);
    sv->data[i] *= (buf + 0.5) + (buf - 0.5) * constr->data[i];
    *psum += pow(creal(sv->data[i]), 2.) + pow(cimag(sv->data[i]), 2.);
  }
  norm = 1 / sqrt(*psum);
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] *= norm;
  }
}

void apply_qpe_diagonals(statevector_t *sv, const diagonals_t *dg,
                         const diagonals_t *constr, const double gamma) {
  cmplx buf;
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    buf = 0.5 * cexp(-I * dg->data[i] * gamma);
    sv->data[i] *= (buf + 0.5) + (buf - 0.5) * constr->data[i];
  }
}

void qpe_qaoa_inner_normalized(statevector_t *sv, const int depth,
                               const diagonals_t *dg, const diagonals_t *constr,
                               double *betas, double *gammas, double *psucc) {
  cmplx val = 1 / sqrt(1 << sv->n_qubits);
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] = val;
  }

  double p_it = 0.;
  *psucc = 1.;
  for (int p = 0; p < depth; p++) {
    apply_qpe_diagonals_normalized(sv, dg, constr, gammas[p], &p_it);
    *psucc *= p_it;
    apply_rx(sv, betas[p]);
  }
}

void qpe_qaoa_inner(statevector_t *sv, const int depth, const diagonals_t *dg,
                    const diagonals_t *constr, const double *betas,
                    const double *gammas, double *psucc) {
  cmplx val = 1 / sqrt(1 << sv->n_qubits);
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] = val;
  }

  for (int p = 0; p < depth; p++) {
    apply_qpe_diagonals(sv, dg, constr, gammas[p]);
    apply_rx(sv, betas[p]);
  }

  *psucc = sv_normalize(sv);
}

statevector_t *qpe_qaoa(const int depth, const diagonals_t *dg,
                        const diagonals_t *constr, double *betas,
                        const double *gammas, double *psucc) {
  statevector_t *sv = sv_malloc(dg->n_qubits);
  qpe_qaoa_inner(sv, depth, dg, constr, betas, gammas, psucc);
  return sv;
}

void grad_qpe_qaoa_inner(int depth, statevector_t *sv_left,
                         statevector_t *sv_right, const diagonals_t *dg,
                         const diagonals_t *cost, const diagonals_t *constr,
                         const double *betas, const double *gammas,
                         double *beta_gradients, double *gamma_gradients,
                         double *psucc, double *expectation_value) {
  qpe_qaoa_inner(sv_left, depth, dg, constr, betas, gammas, psucc);
  for (size_t i = 0; i < 1 << sv_left->n_qubits; i++) {
    sv_left->data[i] *= sqrt(*psucc);
  }
  sv_copy(sv_left, sv_right->data);

  sv_mult(sv_left, cost);
  cmplx buf, gg_buf, dg_buf;
  double dbuf;
  sv_dot(sv_left, sv_right, &buf);

  *expectation_value = creal(buf);
  printf("e: %f\n", *expectation_value);

  for (int p = depth - 1; p >= 0; p--) {
    sum_x_prod(sv_left, sv_right, &beta_gradients[p]);
    sum_x_prod(sv_right, sv_right, &dbuf);
    printf("%f\n", dbuf);
    // beta_gradients[p] -= dbuf / *psucc;
    // beta_gradients[p] /= *psucc;

    apply_rx(sv_left, -betas[p]);
    apply_rx(sv_right, -betas[p]);

    // sv_expec(sv_right, sv_right, dg, &buf);
    // gamma_gradients[p] += 2 * cimag(buf) / *psucc;

    gg_buf = 0;

    for (size_t i = 0; i < 1 << sv_left->n_qubits; i++) {
      dg_buf = 0.5 * cexp(-I * dg->data[i] * gammas[p]);
      cmplx dg_factor = (dg_buf + 0.5) + (dg_buf - 0.5) * constr->data[i];

      cmplx right_appl = sv_right->data[i] / dg_factor;

      cmplx derivative_dg_factor = dg->data[i] * dg_buf * (1 + constr->data[i]);

      gg_buf += conj(sv_left->data[i]) * derivative_dg_factor * right_appl;
      // gg_buf -= conj(sv_right->data[i]) * derivative_dg_factor * right_appl / (*psucc * *psucc);

      sv_right->data[i] = right_appl;
      sv_left->data[i] *= conj(dg_factor);
    }

    gamma_gradients[p] = 2 * cimag(gg_buf);
  }

  sv_print(sv_right);
}

void grad_qpe_qaoa(int depth, const diagonals_t *dg, const diagonals_t *cost,
                   const diagonals_t *constr, const double *betas,
                   const double *gammas, double *beta_gradients,
                   double *gamma_gradients, double *psucc,
                   double *expectation_value) {
  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);
  grad_qpe_qaoa_inner(depth, sv_left, sv_right, dg, cost, constr, betas, gammas,
                      beta_gradients, gamma_gradients, psucc,
                      expectation_value);
  sv_free(sv_left);
  sv_free(sv_right);
}
