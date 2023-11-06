#include <complex.h>

#ifndef _MTYPES
#define _MTYPES

#ifdef USE_FLOAT32
#define real float
#else
#define real double
#endif

#define cmplx real complex

#endif
