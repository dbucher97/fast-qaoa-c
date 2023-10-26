#include "diagonals.h"
#include "statevector.h"

#ifndef _QAOA
#define _QAOA

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
