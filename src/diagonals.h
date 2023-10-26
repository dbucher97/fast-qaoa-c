#include <stdint.h>

#ifndef _DIAGONALS
#define _DIAGONALS

#define TOL 1e-8

typedef struct diagonals_t {
  uint8_t n_qubits;
  double *data;
  double min_val;
  double max_val;
} diagonals_t;;

diagonals_t *dg_malloc(uint8_t n_qubits);

diagonals_t *dg_clone(const diagonals_t *dg);

void dg_copy(const diagonals_t *dg, double *other);

diagonals_t *dg_copy_from(const double *other, const uint8_t n_qubits);

void dg_free(diagonals_t *dg);

void dg_print(const diagonals_t *dg);

diagonals_t *dg_brute_force(uint8_t n_qubits, int polylen, uint64_t *polykeys,
                            double *polyvals);

typedef enum cmp_kind {
  LTE = 0,
  GTE = 1,
  LT = 2,
  GT = 3,
  EQ = 4,
  NEQ = 5
} cmp_kind;

diagonals_t *dg_mask(diagonals_t *dg, diagonals_t *lhs, double rhs,
                     cmp_kind cmp, double val);

diagonals_t *dg_quad_penalty(diagonals_t *dg, diagonals_t *lhs, double rhs,
                             cmp_kind cmp, double *penalty);

diagonals_t *dg_cmp(diagonals_t *lhs, double rhs, cmp_kind cmp);

void dg_shift(diagonals_t *diags, double f);
void dg_scale(diagonals_t *diags, double f);

#endif
