#include "diagonals.h"
#include "statevector.h"

void apply_qpe_diagonals(statevector_t *sv, const diagonals_t *dg,
                         const diagonals_t *constr, const double gamma);

statevector_t *qpe_qaoa(const int depth, const diagonals_t *dg,
                        const diagonals_t *constr, double *params);

void grad_qpe_qaoa(const int depth, const diagonals_t *dg,
                   const diagonals_t *constr, double *params,
                   double *gradients);

void optimize_qpe_qaoa(const int depth, const diagonals_t *dg,
                       const diagonals_t *constr, const double *inital,
                       double *params);
