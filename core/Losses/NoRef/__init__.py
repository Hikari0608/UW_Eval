from .uciqeAndUiqm import calc_uciqe, calc_uiqm
from .niqe import cal_niqe

__all__ = ['calc_uciqe', 'calc_uiqm', 'calc_niqe']

def calc_niqe(x):
    return {'niqe': cal_niqe(x)}