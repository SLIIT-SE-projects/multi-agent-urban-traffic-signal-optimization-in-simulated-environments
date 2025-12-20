import os
import subprocess
import sys
from pathlib import Path

# Configuration
NET_DIR = Path("conf/network")
NET_FILE = NET_DIR / "grid_3x3.net.xml"
ROU_FILE = NET_DIR / "grid_3x3.rou.xml"
CFG_FILE = NET_DIR / "grid_3x3.sumocfg"

def run_command(cmd):
    """Runs a shell command and checks for errors."""
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    # Ensure directory exists
    NET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate Network (3x3 Grid, traffic lights at junctions)
    # --tls.guess: Automatically add traffic lights
    print("--- Generating Network ---")
    cmd_net = [
        "netgenerate",
        "--grid",
        "--grid.number=3",       # 3x3 intersections
        "--grid.length=200",     # 200m blocks
        "--tls.guess",           # Add traffic lights
        f"--output-file={NET_FILE}"
    ]
    run_command(cmd_net)

    # 2. Generate Random Traffic Demand
    # Using SUMO's randomTrips.py tool
    print("--- Generating Traffic Demand ---")
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        print("CRITICAL: SUMO_HOME environment variable not set!")
        sys.exit(1)
        
    random_trips_script = Path(sumo_home) / "tools" / "randomTrips.py"
    
    cmd_trips = [
        "python", str(random_trips_script),
        "-n", str(NET_FILE),
        "-r", str(ROU_FILE),
        "-e", "3600",           # Generate traffic for 1 hour (3600s)
        "-p", "2.0",            # New vehicle every 2.0 seconds (high demand)
        "--random"
    ]
    run_command(cmd_trips)

    # 3. Create SUMO Configuration File (.sumocfg)
    print(f"--- Creating {CFG_FILE} ---")
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{NET_FILE.name}"/>
        <route-files value="{ROU_FILE.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
"""
    with open(CFG_FILE, "w") as f:
        f.write(config_content)

    print("\nSUCCESS: Scenario generated in conf/network/")

if __name__ == "__main__":
    main()