from numba import njit
import numpy as np

@njit
def gpe_ainf(V):
    """
    Steady-state activation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 57) / 2))

@njit
def gpe_minf(V):
    """
    Steady-state activation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 37) / 10))

@njit
def gpe_hinf(V):
    """
    Steady-state inactivation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 58) / 12))

@njit
def gpe_ninf(V):
    """
    Steady-state activation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 50) / 14))


@njit
def gpe_rinf(V):
    """
    Steady-state inactivation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 70) / 2))


@njit
def gpe_sinf(V):
    """
    Steady-state activation function for synaptic input to GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 35) / 2))


@njit
def gpe_tauh(V):
    """
    Time constant function for GPe neuron inactivation.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 0.05 + 0.27 / (1 + np.exp(-(V + 40) / -12))


@njit
def gpe_taun(V):
    """
    Time constant function for GPe neuron activation.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 0.05 + 0.27 / (1 + np.exp(-(V + 40) / -12))


@njit
def Hinf(V):
    """
    Steady-state activation function.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 57) / 2))


@njit
def stn_ainf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 63) / 7.8))


@njit
def stn_binf(R):
    """
    Steady-state activation function for STN neurons based on variable R.

    Parameters:
    R : float or np.ndarray
        Input variable.

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(R - 0.4) / 0.1)) - 1 / (1 + np.exp(0.4 / 0.1))


@njit
def stn_cinf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 20) / 8))


@njit
def stn_hinf(V):
    """
    Steady-state inactivation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 39) / 3.1))


@njit
def stn_minf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 30) / 15))


@njit
def stn_ninf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 32) / 8.0))


@njit
def stn_rinf(V):
    """
    Steady-state inactivation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 67) / 2))


@njit
def stn_sinf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 39) / 8))


@njit
def stn_tauc(V):
    """
    Time constant function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 1 + 10 / (1 + np.exp((V + 80) / 26))


@njit
def stn_tauh(V):
    """
    Time constant function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 1 + 500 / (1 + np.exp(-(V + 57) / -3))


@njit
def stn_taun(V):
    """
    Time constant function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 1 + 100 / (1 + np.exp(-(V + 80) / -26))


@njit
def stn_taur(V):
    """
    Time constant function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 7.1 + 17.5 / (1 + np.exp(-(V - 68) / -2.2))

@njit
def th_hinf(V):
    """
    Steady-state inactivation function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 41) / 4))


@njit
def th_minf(V):
    """
    Computes the steady-state activation (minf) for a thalamic neuron.

    Parameters:
    V (numpy array or float): Membrane potential (in mV)

    Returns:
    numpy array or float: Steady-state activation
    """
    return 1 / (1 + np.exp(-(V + 37) / 7))

@njit
def th_pinf(V):
    """
    Steady-state activation function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 60) / 6.2))

@njit
def th_rinf(V):
    """
    Steady-state inactivation function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 84) / 4))

@njit
def th_tauh(V):
    """
    Time constant function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 1 / (ah(V) + bh(V))

@njit
def ah(V):
    """
    Helper function to compute `a_h` for th_tauh.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
    """
    return 0.128 * np.exp(-(V + 46) / 18)

@njit
def bh(V):
    """
    Helper function to compute `b_h` for th_tauh.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
    """
    return 4 / (1 + np.exp(-(V + 23) / 5))


@njit
def th_taur(V):
    """
    Time constant function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 0.15 * (28 + np.exp(-(V + 25) / 10.5))