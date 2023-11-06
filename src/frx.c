#include "frx.h"
#include <math.h>
#include <stdlib.h>

frx_plan_t *frx_make_plan(statevector_t *sv, Butterfly max_butterfly) {
  frx_plan_t *plan = (frx_plan_t *)malloc(sizeof(frx_plan_t));
  plan->n_qubits = sv->n_qubits;
  plan->n = 1 << plan->n_qubits;
  plan->nh_rdx2 = 1 << (plan->n_qubits - 1);
  plan->nh_rdx4 = 1 << (plan->n_qubits - 2);
  plan->nh_rdx8 = 1 << (plan->n_qubits - 3);

  plan->max_butterfly = max_butterfly;

  uint8_t n_max = plan->n_qubits / max_butterfly;
  uint8_t n_rem = plan->n_qubits % max_butterfly;
  Butterfly bf = max_butterfly - 1;
  Butterfly bf_rem[2];
  uint8_t i = 0;
  while (bf > 0 && n_rem > 0) {
    if (n_rem / bf > 0) {
      bf_rem[i] = bf;
      i++;
      n_rem -= bf;
    }
    bf--;
  }
  plan->n_butterflies = n_max + i;
  plan->butterflies =
      (Butterfly *)malloc(sizeof(Butterfly) * plan->n_butterflies);
  for (uint8_t j = 0; j < n_max; j++) {
    plan->butterflies[j] = max_butterfly;
  }
  for (uint8_t j = 0; j < i; j++) {
    plan->butterflies[j + n_max] = bf_rem[j];
  }

  plan->buffer = (cmplx *)malloc(sizeof(cmplx) * (1 << sv->n_qubits));

  return plan;
}

void frx_free(frx_plan_t *plan) {
  free(plan->butterflies);
  free(plan->buffer);
  free(plan);
}

#define mIt(z) (cimag(z) - I * creal(z))
#define dat(s) data[2 * i + s]
static void apply_bf_rdx2(frx_plan_t *plan, statevector_t *sv) {
  cmplx *b1 = plan->buffer;
  cmplx *b2 = &plan->buffer[plan->nh_rdx2];

  cmplx *data = sv->data;

  for (size_t i = 0; i < plan->nh_rdx2; i++) {
    b1[i] = plan->c * dat(0) + plan->s * mIt(dat(1));
    b2[i] = plan->c * dat(1) + plan->s * mIt(dat(0));
  }

  b1 = sv->data;
  sv->data = plan->buffer;
  plan->buffer = b1;
}
#undef dat

#define dat(s) data[4 * i + s]
static void apply_bf_rdx4(frx_plan_t *plan, statevector_t *sv) {
  cmplx *b1 = plan->buffer;
  cmplx *b2 = &plan->buffer[plan->nh_rdx4];
  cmplx *b3 = &plan->buffer[2 * plan->nh_rdx4];
  cmplx *b4 = &plan->buffer[3 * plan->nh_rdx4];

  cmplx buf;
  cmplx *data = sv->data;
  for (size_t i = 0; i < plan->nh_rdx4; i++) {
    buf = cimag(dat(1)) + cimag(dat(2)) - I * (creal(dat(1)) + creal(dat(2)));
    b1[i] = plan->cc * dat(0) - plan->ss * dat(3) + plan->sc * buf;
    b4[i] = -plan->ss * dat(0) + plan->cc * dat(3) + plan->sc * buf;
    buf = cimag(dat(0)) + cimag(dat(3)) - I * (creal(dat(0)) + creal(dat(3)));
    b2[i] = plan->cc * dat(1) - plan->ss * dat(2) + plan->sc * buf;
    b3[i] = -plan->ss * dat(1) + plan->cc * dat(2) + plan->sc * buf;
  }

  b1 = sv->data;
  sv->data = plan->buffer;
  plan->buffer = b1;
}
#undef dat

#define dat(s) data[8 * i + s]
static void apply_bf_rdx8(frx_plan_t *plan, statevector_t *sv) {
  cmplx *b1 = plan->buffer;
  cmplx *b2 = &plan->buffer[plan->nh_rdx8];
  cmplx *b3 = &plan->buffer[2 * plan->nh_rdx8];
  cmplx *b4 = &plan->buffer[3 * plan->nh_rdx8];
  cmplx *b5 = &plan->buffer[4 * plan->nh_rdx8];
  cmplx *b6 = &plan->buffer[5 * plan->nh_rdx8];
  cmplx *b7 = &plan->buffer[6 * plan->nh_rdx8];
  cmplx *b8 = &plan->buffer[7 * plan->nh_rdx8];
  cmplx bufa, bufb;
  cmplx *data = sv->data;
  for (size_t i = 0; i < plan->nh_rdx8; i++) {
    bufa = dat(1) + dat(2) + dat(4);
    bufb = dat(3) + dat(5) + dat(6);
    b1[i] = plan->ccc * dat(0) + plan->sss * mIt(dat(7)) - plan->ssc * bufb +
            plan->scc * mIt(bufa);
    b8[i] = plan->ccc * dat(7) + plan->sss * mIt(dat(0)) - plan->ssc * bufa +
            plan->scc * mIt(bufb);
    bufa += dat(7) - dat(1);
    bufb += dat(0) - dat(6);
    b2[i] = plan->ccc * dat(1) + plan->sss * mIt(dat(6)) - plan->ssc * bufa +
            plan->scc * mIt(bufb);
    b7[i] = plan->ccc * dat(6) + plan->sss * mIt(dat(1)) - plan->ssc * bufb +
            plan->scc * mIt(bufa);
    bufa += dat(1) - dat(2);
    bufb += dat(6) - dat(5);
    b3[i] = plan->ccc * dat(2) + plan->sss * mIt(dat(5)) - plan->ssc * bufa +
            plan->scc * mIt(bufb);
    b6[i] = plan->ccc * dat(5) + plan->sss * mIt(dat(2)) - plan->ssc * bufb +
            plan->scc * mIt(bufa);
    bufa += dat(2) - dat(4);
    bufb += dat(5) - dat(3);
    b4[i] = plan->ccc * dat(3) + plan->sss * mIt(dat(4)) - plan->ssc * bufb +
            plan->scc * mIt(bufa);
    b5[i] = plan->ccc * dat(4) + plan->sss * mIt(dat(3)) - plan->ssc * bufa +
            plan->scc * mIt(bufb);
  }

  b1 = sv->data;
  sv->data = plan->buffer;
  plan->buffer = b1;
}
#undef dat

void frx_apply(frx_plan_t *plan, statevector_t *sv, real beta) {
  plan->s = sin(beta);
  plan->c = cos(beta);
  if (plan->max_butterfly >= RDX4) {
    plan->ss = plan->s * plan->s;
    plan->cc = 1 - plan->ss;
    plan->sc = plan->s * plan->c;
  }
  if (plan->max_butterfly >= RDX8) {
    plan->ssc = plan->ss * plan->c;
    plan->scc = plan->cc * plan->s;
    plan->sss = plan->ss * plan->s;
    plan->ccc = plan->cc * plan->c;
  }

  int q = 0;

  for (size_t i = 0; i < plan->n_butterflies; i++) {
    switch (plan->butterflies[i]) {
    case RDX2:
      apply_bf_rdx2(plan, sv);
      break;
      ;
    case RDX4:
      apply_bf_rdx4(plan, sv);
      break;
      ;
    case RDX8:
      apply_bf_rdx8(plan, sv);
      break;
      ;
    }
    q += plan->butterflies[i];
  }
}
