#include "diagonals.h"
#include "frx.h"
#include "statevector.h"

#ifndef _QPE_QAOA
#define _QPE_QAOA

void apply_qpe_diagonals_normalized(statevector_t *sv, const diagonals_t *dg,
                                    const diagonals_t *constr,
                                    const real gamma, real *psum);

void apply_qpe_diagonals(statevector_t *sv, const diagonals_t *dg,
                         const diagonals_t *constr, const real gamma);

void qpe_qaoa_inner(statevector_t *sv, frx_plan_t *frx, const int depth,
                    const diagonals_t *dg, const diagonals_t *constr,
                    const real *betas, const real *gammas, real *psucc);

statevector_t *qpe_qaoa(const int depth, const diagonals_t *dg,
                        const diagonals_t *constr, real *betas,
                        const real *gammas, real *psucc);

void grad_qpe_qaoa_inner(statevector_t *sv_left, statevector_t *sv_right,
                         statevector_t *sv_left_p, frx_plan_t *plan, int depth,
                         const diagonals_t *dg, const diagonals_t *cost,
                         const diagonals_t *constr, const real *betas,
                         const real *gammas, real *beta_gradients,
                         real *gamma_gradients, real *psucc,
                         real *expectation_value);

void grad_qpe_qaoa(int depth, const diagonals_t *dg, const diagonals_t *cost,
                   const diagonals_t *constr, const real *betas,
                   const real *gammas, real *beta_gradients,
                   real *gamma_gradients, real *psucc,
                   real *expectation_value);

#endif
