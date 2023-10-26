#include "diagonals.h"

diagonals_t* ind_compute_single_delta(uint8_t n_qubits, double offset);

diagonals_t* ind_compute_combined(uint8_t n_qubits, double offset);

diagonals_t* ind_interpolate(diagonals_t* indicator, diagonals_t* constr_diag);
