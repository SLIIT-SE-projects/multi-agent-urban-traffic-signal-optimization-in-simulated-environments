import os
import sys
from pathlib import Path

# Ensure we can find sumolib
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import sumolib

def generate_e2_detectors(net_file, output_file):
    print(f"Reading network: {net_file}")
    net = sumolib.net.readNet(net_file)
    
    with open(output_file, "w") as f:
        f.write('<additional>\n')
        
        # Iterate over all edges (roads)
        for edge in net.getEdges():
            # Skip internal SUMO edges (functions of intersections)
            if edge.getFunction() == 'internal':
                continue
                
            for lane in edge.getLanes():
                lane_id = lane.getID()
                length = lane.getLength()
                
                # We place a detector covering the last 80% of the lane (typical for queue detection)
                # pos: start position, length: length of the detector box
                # E2 Detectors measure "Jam Length" directly, which is critical for MPC.
                det_id = f"e2_{lane_id}"
                
                # XML definition for E2 detector
                f.write(f'    <laneAreaDetector id="{det_id}" lane="{lane_id}" '
                        f'pos="0" length="{length}" freq="1.0" '
                        f'file="e2_output.xml"/>\n')
                        
        f.write('</additional>\n')
    print(f"Successfully generated detectors: {output_file}")

if __name__ == "__main__":
    NET_FILE = Path("conf/network/grid_3x3.net.xml")
    ADD_FILE = Path("conf/network/grid_3x3.add.xml")
    
    if not NET_FILE.exists():
        print(f"Error: Network file {NET_FILE} not found. Run generate_scenario.py first.")
    else:
        generate_e2_detectors(str(NET_FILE), str(ADD_FILE))