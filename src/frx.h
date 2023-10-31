#include <stdint.h>
#include "cmplx.h"
#include "statevector.h"

#ifndef _FRX
#define _FRX

typedef enum Butterfly {
    // FWHT = 0,
    RDX2 = 1,
    RDX4 = 2,
    RDX8 = 3,
} Butterfly;


typedef struct frx_plan_t {
    uint8_t n_butterflies;
    uint8_t n_qubits;
    size_t n, nh_rdx2, nh_rdx4, nh_rdx8;
    Butterfly max_butterfly;
    Butterfly* butterflies;
    cmplx* buffer;
    double c, cc, ccc, s, ss, sss, sc, ssc, scc;
} frx_plan_t;

frx_plan_t* frx_make_plan(statevector_t* sv, Butterfly max_butterfly);

void frx_free(frx_plan_t* plan);

void frx_apply(frx_plan_t* plan, statevector_t* sv, double beta);

#endif


