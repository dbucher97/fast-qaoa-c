#include "diagonals.h"
#include "statevector.h"

#ifndef _QPE_QAOA
#define _QPE_QAOA

void apply_qpe_diagonals_normalized(statevector_t *sv, const diagonals_t *dg,
                                    const diagonals_t *constr,
                                    const double gamma, double *psum);

void apply_qpe_diagonals(statevector_t *sv, const diagonals_t *dg,
                         const diagonals_t *constr, const double gamma);

void qpe_qaoa_inner(statevector_t *sv, const int depth, const diagonals_t *dg,
                    const diagonals_t *constr, const double *betas,
                    const double *gammas, double *psucc);

statevector_t *qpe_qaoa(const int depth, const diagonals_t *dg,
                        const diagonals_t *constr, double *betas,
                        const double *gammas, double *psucc);

void grad_qpe_qaoa_inner(int depth, statevector_t *sv_left,
                         statevector_t *sv_right, const diagonals_t *dg,
                         const diagonals_t *cost, const diagonals_t *constr,
                         const double *betas, const double *gammas,
                         double *beta_gradients, double *gamma_gradients,
                         double *psucc, double *expectation_value);

void grad_qpe_qaoa(int depth, const diagonals_t *dg, const diagonals_t *cost,
                   const diagonals_t *constr, const double *betas,
                   const double *gammas, double *beta_gradients,
                   double *gamma_gradients, double *psucc,
                   double *expectation_value);

#endif
