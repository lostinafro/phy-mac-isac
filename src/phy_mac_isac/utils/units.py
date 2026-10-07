# Units & numeric transformations: dB, dBm, Watt, rounding
import numpy as np


def lin2db(lin):
    return 10 * np.log10(lin)


def db2lin(db):
    return 10 ** (db / 10)


def dbm2watt(dbm):
    """Simply converts dBm to Watt"""
    return 10 ** (dbm / 10 - 3)


def watt2dbm(watt):
    """Simply converts Watt to dBm"""
    return 10 * np.log10(watt * 1e3)


def np_ceil(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)