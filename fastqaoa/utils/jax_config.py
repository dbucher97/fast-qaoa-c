import importlib
from enum import Enum

if importlib.util.find_spec("jax") is not None:
    import jax
    import jax.numpy as jnp

class Accuracy(int, Enum):
    single = 32
    double = 64

_accuracy = Accuracy.single

def get_accuracy():
    return _accuracy

def set_accuracy(accuracy: Accuracy):
    accuracy = Accuracy(accuracy)
    global _accuracy
    if _accuracy == accuracy:
        return
    elif accuracy == Accuracy.double:
        jax.config.update("jax_enable_x64", True)
        _accuracy = accuracy
    elif accuracy == Accuracy.single:
        jax.config.update("jax_enable_x64", False)
        _accuracy = accuracy


def get_real_dtype():
    if _accuracy == Accuracy.double:
        return jnp.float64
    else:
        return jnp.float32

def get_complex_dtype():
    if _accuracy == Accuracy.double:
        return jnp.complex128
    else:
        return jnp.complex64

