import sys
import os
import casadi as ca
import pydantic
import traci

from importlib.metadata import version

def check_setup():
    print("--- Environment Verification ---")
    
    # 1. Check Python Version
    print(f"Python Version: {sys.version.split()[0]}")
    
    # 2. Check CasADi (The Solver)
    print(f"CasADi Version: {ca.__version__}")
    opti = ca.Opti()
    x = opti.variable()
    opti.minimize(x**2)
    opti.subject_to(x >= 2)
    opti.solver('ipopt', {'print_time': 0, 'ipopt.print_level': 0})
    sol = opti.solve()
    print(f"CasADi Test Optimization (min x^2 s.t. x>=2): Result = {sol.value(x)} (Expected: 2.0)")

    # 3. Check Pydantic (Configuration)
    print(f"Pydantic Version: {pydantic.VERSION}")

    # 4. Check SUMO/TraCI
    if 'SUMO_HOME' in os.environ:
        print(f"SUMO_HOME found: {os.environ['SUMO_HOME']}")
        print(f"TraCI Version: {version('traci')}")
    else:
        print("CRITICAL WARNING: SUMO_HOME environment variable is NOT set.")

if __name__ == "__main__":
    check_setup()