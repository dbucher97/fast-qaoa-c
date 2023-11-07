#include <math.h>
#ifndef _M_MATH
#define _M_MATH

#ifdef USE_FLOAT32
#define COS cosf
#define SIN sinf
#else
#define COS cos
#define SIN sin
#endif

#endif
