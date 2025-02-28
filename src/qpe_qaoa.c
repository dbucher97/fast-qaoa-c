#include "qpe_qaoa.h"
#include "qaoa.h"
#include "mmath.h"
#include <math.h>
#include <stdio.h>

#define EPS 1e-6

void apply_qpe_diagonals_normalized(statevector_t *sv, const diagonals_t *dg,
                                    const diagonals_t *constr, const real gamma,
                                    real *psum) {
  cmplx buf;
  real norm;
  *psum = 0;
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    buf = 0.5 * cexp(I * dg->data[i] * gamma);
    sv->data[i] *= (buf + 0.5) + (buf - 0.5) * constr->data[i];
    *psum += pow(creal(sv->data[i]), 2.) + pow(cimag(sv->data[i]), 2.);
  }
  norm = 1 / sqrt(*psum);
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] *= norm;
  }
}

void apply_qpe_diagonals(statevector_t *sv, const diagonals_t *dg,
                         const diagonals_t *constr, const real gamma) {
  cmplx buf;
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    buf = 0.5 * cexp(I * dg->data[i] * gamma);
    sv->data[i] *= (buf + 0.5) + (buf - 0.5) * constr->data[i];
  }
}

void qpe_qaoa_inner_normalized(statevector_t *sv, frx_plan_t *plan,
                               const int depth, const diagonals_t *dg,
                               const diagonals_t *constr, const real *betas,
                               const real *gammas, real *psucc,
                               real *psucc_array) {
  cmplx val = 1 / sqrt(1 << sv->n_qubits);
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] = val;
  }

  real p_it = 0.;
  *psucc = 1.;
  for (int p = 0; p < depth; p++) {
    apply_qpe_diagonals_normalized(sv, dg, constr, gammas[p], &p_it);
    *psucc *= p_it;
    if (psucc_array != NULL) {
      psucc_array[p] = p_it;
    }
    frx_apply(plan, sv, betas[p]);
  }
}

void qpe_qaoa_inner(statevector_t *sv, frx_plan_t *plan, const int depth,
                    const diagonals_t *dg, const diagonals_t *constr,
                    const real *betas, const real *gammas, real *psucc) {
  cmplx val = 1 / sqrt(1 << sv->n_qubits);
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] = val;
  }

  for (int p = 0; p < depth; p++) {
    apply_qpe_diagonals(sv, dg, constr, gammas[p]);
    frx_apply(plan, sv, betas[p]);
  }

  *psucc = sv_normalize(sv);
}

statevector_t *qpe_qaoa(const int depth, const diagonals_t *dg,
                        const diagonals_t *constr, real *betas,
                        const real *gammas, real *psucc) {
  statevector_t *sv = sv_malloc(dg->n_qubits);
  frx_plan_t *plan = frx_make_plan(sv, RDX4);

  qpe_qaoa_inner(sv, plan, depth, dg, constr, betas, gammas, psucc);

  frx_free(plan);
  return sv;
}

statevector_t *qpe_qaoa_norm(const int depth, const diagonals_t *dg,
                             const diagonals_t *constr, real *betas,
                             const real *gammas, real *psucc,
                             real *psucc_array) {
  statevector_t *sv = sv_malloc(dg->n_qubits);
  frx_plan_t *plan = frx_make_plan(sv, RDX4);

  qpe_qaoa_inner_normalized(sv, plan, depth, dg, constr, betas, gammas, psucc,
                            psucc_array);

  frx_free(plan);
  return sv;
}

void grad_qpe_qaoa_inner(statevector_t *sv_left, statevector_t *sv_right,
                         statevector_t *sv_left_p, frx_plan_t *plan, int depth,
                         const diagonals_t *dg, const diagonals_t *cost,
                         const diagonals_t *constr, const real *betas,
                         const real *gammas, real *beta_gradients,
                         real *gamma_gradients, real *psucc,
                         real *expectation_value) {
  cmplx buf, gg_buf, dg_buf, dg_factor, right_appl, derivative_dg_factor;
  real dbuf;

  qpe_qaoa_inner(sv_left, plan, depth, dg, constr, betas, gammas, psucc);
  sv_copy(sv_left, sv_right->data);
  sv_copy(sv_left, sv_left_p->data);

  sv_mult(sv_left, cost);
  sv_dot(sv_left, sv_right, &buf);
  *expectation_value = creal(buf);

  for (int p = depth - 1; p >= 0; p--) {
    sum_x_prod(sv_left, sv_right, &beta_gradients[p]);
    sum_x_prod(sv_left_p, sv_right, &dbuf);
    beta_gradients[p] -= *expectation_value * dbuf;

    frx_apply(plan, sv_left, -betas[p]);
    frx_apply(plan, sv_right, -betas[p]);
    frx_apply(plan, sv_left_p, -betas[p]);

    gg_buf = 0;
    for (size_t i = 0; i < 1 << sv_left->n_qubits; i++) {
      dg_buf = 0.5 * cexp(I * dg->data[i] * gammas[p]);
      dg_factor = (dg_buf + 0.5) + (dg_buf - 0.5) * constr->data[i];

      right_appl = sv_right->data[i] / dg_factor;

      derivative_dg_factor =
          dg->data[i] * dg_buf * (1 + constr->data[i]) * right_appl;

      gg_buf += conj(sv_left->data[i]) * derivative_dg_factor;
      gg_buf -=
          *expectation_value * conj(sv_left_p->data[i]) * derivative_dg_factor;

      sv_right->data[i] = right_appl;
      sv_left->data[i] *= conj(dg_factor);
      sv_left_p->data[i] *= conj(dg_factor);
    }

    gamma_gradients[p] = -2. * cimag(gg_buf);
  }
}

void grad_qpe_qaoa(int depth, const diagonals_t *dg, const diagonals_t *cost,
                   const diagonals_t *constr, const real *betas,
                   const real *gammas, real *beta_gradients,
                   real *gamma_gradients, real *psucc,
                   real *expectation_value) {
  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);
  statevector_t *sv_left_p = sv_malloc(dg->n_qubits);
  frx_plan_t *plan = frx_make_plan(sv_left, RDX4);

  grad_qpe_qaoa_inner(sv_left, sv_right, sv_left_p, plan, depth, dg, cost,
                      constr, betas, gammas, beta_gradients, gamma_gradients,
                      psucc, expectation_value);
  frx_free(plan);
  sv_free(sv_left);
  sv_free(sv_left_p);
  sv_free(sv_right);
}
