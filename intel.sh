#! /usr/bin/env sh

ONEAPI=/opt/intel/oneapi
source $ONEAPI/setvars.sh

ICX=$ONEAPI/compiler/latest/linux
MKL=$ONEAPI/mkl/latest

export CFLAGS="-I$MKL/include -I$ICX/include"
export LDFLAGS="-L$MKL/lib -L$ICX/lib"

export CC="$ICX/bin/icx"
