import sys
print sys.path
import rk4.cytests as ct

def test_ode():
    return ct.test_ode()

def test_create_ode_callback():
    return ct.test_create_ode_callback()

def test_ode_callback():
    return ct.test_ode_callback()

def test_sample_gaussian():
    return ct.test_sample_gaussian()

def test_sde():
    return ct.test_sde()

def test_sde_callback():
    return ct.test_sde_callback()