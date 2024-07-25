#! /usr/bin/env sh

ONEAPI=/home/bucher/intel/oneapi
source $ONEAPI/setvars.sh

ICX=$ONEAPI/compiler/latest
MKL=$ONEAPI/mkl/latest

export CFLAGS="-I$MKL/include -I$ICX/include"
export LDFLAGS="-L$MKL/lib -L$ICX/lib"

export CC="$ICX/bin/icx"
