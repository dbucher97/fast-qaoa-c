#include "qaoa.h"
#include "frx.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>

void apply_diagonals(statevector_t *sv, const diagonals_t *dg,
                     const double gamma) {
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] *= cos(dg->data[i] * gamma) - I * sin(dg->data[i] * gamma);
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

void apply_rx_old(statevector_t *sv, const double beta) {
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

void apply_rx(statevector_t *sv, const double beta) {
  frx_plan_t* plan = frx_make_plan(sv, RDX4);
  frx_apply(plan, sv, beta);
  frx_free(plan);
}

// void apply_rx(statevector_t *sv, const double beta) {
//   // cmplx facs[sv->n_qubits + 1];
//   // double norm = 1 / (double)(1 << sv->n_qubits);
//   // facs[0] = cexp(I * beta * sv->n_qubits);
//   // cmplx delta = cexp(-2 * I * beta);
//   // for (uint8_t i = 1; i <= sv->n_qubits; i++) {
//   //   facs[i] = facs[0] * delta;
//   // }
//   // for (uint8_t i = 0; i < sv->n_qubits; i++) {
//   //   facs[i] *= norm;
//   // }
//
//   cmplx *buf1 = (cmplx*) malloc(sizeof(cmplx) * 1 << sv->n_qubits);
//   cmplx *buf2 = sv->data;
//   cmplx *tmp;
//
//   cmplx s = -I * 1.346309;
//   cmplx c = 1.31346609;
//
//   const size_t nqh = 1 << (sv->n_qubits -1);
//   for (uint8_t q = 0; q < sv->n_qubits; q++) {
//     cblas_zscal(sv->n_qubits, &c, buf2, 1);
//     cblas_zcopy(nqh, buf2, 2, buf1, 1);
//     cblas_zcopy(nqh, &buf2[1], 2, &buf1[nqh], 1);
//     cblas_zaxpy(nqh, &s, buf2, 2, buf1, 1);
//     cblas_zaxpy(nqh, &s, &buf2[1], 2, &buf1[nqh], 1);
//     tmp = buf2; buf2 = buf1; buf1 = tmp;
//   }
//
//   free(buf1);
//
//   // size_t u;
//   // for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
//   //   u = __builtin_popcount(i);
//   //   data[i] *= facs[u];
//   // }
//
//   // for (uint8_t q = 0; q < sv->n_qubits; q++) {
//   //   for (size_t i1 = 0; i1 < (1 << (sv->n_qubits)); i1 += 1 << (q + 1)) {
//   //     for (size_t i2 = i1; i2 < i1 + (1 << q); i2++) {
//   //       cmplx a = data[i2];
//   //       cmplx b = data[i2 + (1 << q)];
//   //
//   //       data[i2] = a + b;
//   //       data[i2 + (1 << q)] = a - b;
//   //     }
//   //   }
//   // }
// }

void qaoa_inner(statevector_t *sv, frx_plan_t* plan, const int depth, const diagonals_t *dg,
                const double *betas, const double *gammas) {
  cmplx val = 1 / sqrt(1 << sv->n_qubits);
  for (size_t i = 0; i < 1 << sv->n_qubits; i++) {
    sv->data[i] = val;
  }

  for (int p = 0; p < depth; p++) {
    apply_diagonals(sv, dg, gammas[p]);
    // apply_rx(sv, betas[p]);
    frx_apply(plan, sv, betas[p]);
  }
}

statevector_t *qaoa(const int depth, const diagonals_t *dg, const double *betas,
                    const double *gammas) {
  statevector_t *sv = sv_malloc(dg->n_qubits);
  frx_plan_t* plan = frx_make_plan(sv, RDX4);

  qaoa_inner(sv, plan, depth, dg, betas, gammas);

  frx_free(plan);
  return sv;
}

void grad_qaoa_inner(statevector_t *sv_left, statevector_t *sv_right, frx_plan_t* plan,
                     const int depth, const diagonals_t *dg,
                     const diagonals_t *cost, const double *betas,
                     const double *gammas, double *beta_gradients,
                     double *gamma_gradients, double *expectation_value) {
  qaoa_inner(sv_left, plan, depth, dg, betas, gammas);
  sv_copy(sv_left, sv_right->data);

  sv_mult(sv_left, cost);

  cmplx buf;
  sv_dot(sv_left, sv_right, &buf);
  *expectation_value = creal(buf);


  for (int p = depth - 1; p >= 0; p--) {
    sum_x_prod(sv_left, sv_right, &beta_gradients[p]);

    frx_apply(plan, sv_left, -betas[p]);
    frx_apply(plan, sv_right, -betas[p]);

    sv_expec(sv_left, sv_right, dg, &buf);
    gamma_gradients[p] = 2 * cimag(buf);

    for (size_t i = 0; i < 1 << sv_left->n_qubits; i++) {
      buf = cexp(I * dg->data[i] * gammas[p]);
      sv_left->data[i] *= buf;
      sv_right->data[i] *= buf;
    }
  }

}

double grad_qaoa(const int depth, const diagonals_t *dg,
                 const diagonals_t *cost, const double *betas,
                 const double *gammas, double *beta_gradients,
                 double *gamma_gradients) {
  statevector_t *sv_left = sv_malloc(dg->n_qubits);
  statevector_t *sv_right = sv_malloc(dg->n_qubits);
  frx_plan_t* plan = frx_make_plan(sv_left, RDX4);

  double expectation_value;
  grad_qaoa_inner(sv_left, sv_right, plan, depth, dg, cost, betas, gammas,
                  beta_gradients, gamma_gradients, &expectation_value);

  sv_free(sv_left);
  sv_free(sv_right);
  frx_free(plan);

  return expectation_value;
}
