"""
Create a 3x3 grid SUMO network with LEFT-HAND TRAFFIC (for Sri Lanka, UK, India, etc.)
Run this from your TrafficProject root directory
"""
import os
import subprocess

# Create directory
output_dir = os.path.join("scenarios", "grid3x3")
os.makedirs(output_dir, exist_ok=True)
print(f"✓ Created directory: {output_dir}")

# File contents dictionary (same as before)
files = {
    "grid3x3.nod.xml": '''<?xml version="1.0" encoding="UTF-8"?>
<!-- 3x3 Grid with 9 signalized intersections -->
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <!-- 9 Signalized Intersections (3x3 grid) -->
    <node id="J00" x="0.0" y="0.0" type="traffic_light"/>
    <node id="J01" x="0.0" y="250.0" type="traffic_light"/>
    <node id="J02" x="0.0" y="500.0" type="traffic_light"/>
    
    <node id="J10" x="250.0" y="0.0" type="traffic_light"/>
    <node id="J11" x="250.0" y="250.0" type="traffic_light"/>
    <node id="J12" x="250.0" y="500.0" type="traffic_light"/>
    
    <node id="J20" x="500.0" y="0.0" type="traffic_light"/>
    <node id="J21" x="500.0" y="250.0" type="traffic_light"/>
    <node id="J22" x="500.0" y="500.0" type="traffic_light"/>
    
    <!-- Boundary nodes - West (left side) -->
    <node id="W0" x="-250.0" y="0.0" type="priority"/>
    <node id="W1" x="-250.0" y="250.0" type="priority"/>
    <node id="W2" x="-250.0" y="500.0" type="priority"/>
    
    <!-- Boundary nodes - East (right side) -->
    <node id="E0" x="750.0" y="0.0" type="priority"/>
    <node id="E1" x="750.0" y="250.0" type="priority"/>
    <node id="E2" x="750.0" y="500.0" type="priority"/>
    
    <!-- Boundary nodes - South (bottom) -->
    <node id="S0" x="0.0" y="-250.0" type="priority"/>
    <node id="S1" x="250.0" y="-250.0" type="priority"/>
    <node id="S2" x="500.0" y="-250.0" type="priority"/>
    
    <!-- Boundary nodes - North (top) -->
    <node id="N0" x="0.0" y="750.0" type="priority"/>
    <node id="N1" x="250.0" y="750.0" type="priority"/>
    <node id="N2" x="500.0" y="750.0" type="priority"/>
</nodes>''',

    "grid3x3.edg.xml": '''<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- HORIZONTAL EDGES - West to East (Left to Right) -->
    <!-- Row 0 (bottom) -->
    <edge id="W0_J00" from="W0" to="J00" numLanes="2" speed="13.89"/>
    <edge id="J00_J10" from="J00" to="J10" numLanes="2" speed="13.89"/>
    <edge id="J10_J20" from="J10" to="J20" numLanes="2" speed="13.89"/>
    <edge id="J20_E0" from="J20" to="E0" numLanes="2" speed="13.89"/>
    
    <!-- Row 1 (middle) -->
    <edge id="W1_J01" from="W1" to="J01" numLanes="2" speed="13.89"/>
    <edge id="J01_J11" from="J01" to="J11" numLanes="2" speed="13.89"/>
    <edge id="J11_J21" from="J11" to="J21" numLanes="2" speed="13.89"/>
    <edge id="J21_E1" from="J21" to="E1" numLanes="2" speed="13.89"/>
    
    <!-- Row 2 (top) -->
    <edge id="W2_J02" from="W2" to="J02" numLanes="2" speed="13.89"/>
    <edge id="J02_J12" from="J02" to="J12" numLanes="2" speed="13.89"/>
    <edge id="J12_J22" from="J12" to="J22" numLanes="2" speed="13.89"/>
    <edge id="J22_E2" from="J22" to="E2" numLanes="2" speed="13.89"/>
    
    <!-- HORIZONTAL EDGES - East to West (Right to Left) -->
    <!-- Row 0 (bottom) -->
    <edge id="E0_J20" from="E0" to="J20" numLanes="2" speed="13.89"/>
    <edge id="J20_J10" from="J20" to="J10" numLanes="2" speed="13.89"/>
    <edge id="J10_J00" from="J10" to="J00" numLanes="2" speed="13.89"/>
    <edge id="J00_W0" from="J00" to="W0" numLanes="2" speed="13.89"/>
    
    <!-- Row 1 (middle) -->
    <edge id="E1_J21" from="E1" to="J21" numLanes="2" speed="13.89"/>
    <edge id="J21_J11" from="J21" to="J11" numLanes="2" speed="13.89"/>
    <edge id="J11_J01" from="J11" to="J01" numLanes="2" speed="13.89"/>
    <edge id="J01_W1" from="J01" to="W1" numLanes="2" speed="13.89"/>
    
    <!-- Row 2 (top) -->
    <edge id="E2_J22" from="E2" to="J22" numLanes="2" speed="13.89"/>
    <edge id="J22_J12" from="J22" to="J12" numLanes="2" speed="13.89"/>
    <edge id="J12_J02" from="J12" to="J02" numLanes="2" speed="13.89"/>
    <edge id="J02_W2" from="J02" to="W2" numLanes="2" speed="13.89"/>
    
    <!-- VERTICAL EDGES - South to North (Bottom to Top) -->
    <!-- Column 0 (left) -->
    <edge id="S0_J00" from="S0" to="J00" numLanes="2" speed="13.89"/>
    <edge id="J00_J01" from="J00" to="J01" numLanes="2" speed="13.89"/>
    <edge id="J01_J02" from="J01" to="J02" numLanes="2" speed="13.89"/>
    <edge id="J02_N0" from="J02" to="N0" numLanes="2" speed="13.89"/>
    
    <!-- Column 1 (middle) -->
    <edge id="S1_J10" from="S1" to="J10" numLanes="2" speed="13.89"/>
    <edge id="J10_J11" from="J10" to="J11" numLanes="2" speed="13.89"/>
    <edge id="J11_J12" from="J11" to="J12" numLanes="2" speed="13.89"/>
    <edge id="J12_N1" from="J12" to="N1" numLanes="2" speed="13.89"/>
    
    <!-- Column 2 (right) -->
    <edge id="S2_J20" from="S2" to="J20" numLanes="2" speed="13.89"/>
    <edge id="J20_J21" from="J20" to="J21" numLanes="2" speed="13.89"/>
    <edge id="J21_J22" from="J21" to="J22" numLanes="2" speed="13.89"/>
    <edge id="J22_N2" from="J22" to="N2" numLanes="2" speed="13.89"/>
    
    <!-- VERTICAL EDGES - North to South (Top to Bottom) -->
    <!-- Column 0 (left) -->
    <edge id="N0_J02" from="N0" to="J02" numLanes="2" speed="13.89"/>
    <edge id="J02_J01" from="J02" to="J01" numLanes="2" speed="13.89"/>
    <edge id="J01_J00" from="J01" to="J00" numLanes="2" speed="13.89"/>
    <edge id="J00_S0" from="J00" to="S0" numLanes="2" speed="13.89"/>
    
    <!-- Column 1 (middle) -->
    <edge id="N1_J12" from="N1" to="J12" numLanes="2" speed="13.89"/>
    <edge id="J12_J11" from="J12" to="J11" numLanes="2" speed="13.89"/>
    <edge id="J11_J10" from="J11" to="J10" numLanes="2" speed="13.89"/>
    <edge id="J10_S1" from="J10" to="S1" numLanes="2" speed="13.89"/>
    
    <!-- Column 2 (right) -->
    <edge id="N2_J22" from="N2" to="J22" numLanes="2" speed="13.89"/>
    <edge id="J22_J21" from="J22" to="J21" numLanes="2" speed="13.89"/>
    <edge id="J21_J20" from="J21" to="J20" numLanes="2" speed="13.89"/>
    <edge id="J20_S2" from="J20" to="S2" numLanes="2" speed="13.89"/>
</edges>''',

    "grid3x3.rou.xml": '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Vehicle Types -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50" color="1,1,0"/>
    <vType id="bus" accel="1.2" decel="4.0" sigma="0.3" length="12" maxSpeed="40" vClass="bus" color="0,1,1"/>
    <vType id="truck" accel="1.8" decel="4.0" sigma="0.4" length="8" maxSpeed="40" color="0.5,0.5,0.5"/>
    <vType id="emergency" accel="3.5" decel="5.5" sigma="0.2" length="6" maxSpeed="60" vClass="emergency" color="1,0,0"/>
    
    <!-- HORIZONTAL ROUTES (West to East) -->
    <route id="route_WE_0" edges="W0_J00 J00_J10 J10_J20 J20_E0"/>
    <route id="route_WE_1" edges="W1_J01 J01_J11 J11_J21 J21_E1"/>
    <route id="route_WE_2" edges="W2_J02 J02_J12 J12_J22 J22_E2"/>
    
    <!-- HORIZONTAL ROUTES (East to West) -->
    <route id="route_EW_0" edges="E0_J20 J20_J10 J10_J00 J00_W0"/>
    <route id="route_EW_1" edges="E1_J21 J21_J11 J11_J01 J01_W1"/>
    <route id="route_EW_2" edges="E2_J22 J22_J12 J12_J02 J02_W2"/>
    
    <!-- VERTICAL ROUTES (South to North) -->
    <route id="route_SN_0" edges="S0_J00 J00_J01 J01_J02 J02_N0"/>
    <route id="route_SN_1" edges="S1_J10 J10_J11 J11_J12 J12_N1"/>
    <route id="route_SN_2" edges="S2_J20 J20_J21 J21_J22 J22_N2"/>
    
    <!-- VERTICAL ROUTES (North to South) -->
    <route id="route_NS_0" edges="N0_J02 J02_J01 J01_J00 J00_S0"/>
    <route id="route_NS_1" edges="N1_J12 J12_J11 J11_J10 J10_S1"/>
    <route id="route_NS_2" edges="N2_J22 J22_J21 J21_J20 J20_S2"/>
    
    <!-- DIAGONAL ROUTES - FIXED -->
    <route id="route_diagonal_1" edges="W0_J00 J00_J01 J01_J11 J11_J21 J21_E1"/>
    <route id="route_diagonal_2" edges="S0_J00 J00_J10 J10_J11 J11_J12 J12_N1"/>
    <route id="route_diagonal_3" edges="E0_J20 J20_J10 J10_J11 J11_J12 J12_J02 J02_N0"/>
    <route id="route_diagonal_4" edges="S2_J20 J20_J21 J21_J11 J11_J01 J01_W1"/>
    
    <!-- TRAFFIC FLOWS -->
    <flow id="flow_WE_0" type="car" route="route_WE_0" begin="0" end="3600" vehsPerHour="300" departLane="best"/>
    <flow id="flow_WE_1" type="car" route="route_WE_1" begin="0" end="3600" vehsPerHour="450" departLane="best"/>
    <flow id="flow_WE_2" type="car" route="route_WE_2" begin="0" end="3600" vehsPerHour="300" departLane="best"/>
    
    <flow id="flow_EW_0" type="car" route="route_EW_0" begin="0" end="3600" vehsPerHour="320" departLane="best"/>
    <flow id="flow_EW_1" type="car" route="route_EW_1" begin="0" end="3600" vehsPerHour="480" departLane="best"/>
    <flow id="flow_EW_2" type="car" route="route_EW_2" begin="0" end="3600" vehsPerHour="320" departLane="best"/>
    
    <flow id="flow_SN_0" type="car" route="route_SN_0" begin="0" end="3600" vehsPerHour="280" departLane="best"/>
    <flow id="flow_SN_1" type="car" route="route_SN_1" begin="0" end="3600" vehsPerHour="350" departLane="best"/>
    <flow id="flow_SN_2" type="car" route="route_SN_2" begin="0" end="3600" vehsPerHour="280" departLane="best"/>
    
    <flow id="flow_NS_0" type="car" route="route_NS_0" begin="0" end="3600" vehsPerHour="260" departLane="best"/>
    <flow id="flow_NS_1" type="car" route="route_NS_1" begin="0" end="3600" vehsPerHour="330" departLane="best"/>
    <flow id="flow_NS_2" type="car" route="route_NS_2" begin="0" end="3600" vehsPerHour="260" departLane="best"/>
    
    <flow id="flow_diag_1" type="car" route="route_diagonal_1" begin="0" end="3600" vehsPerHour="120" departLane="best"/>
    <flow id="flow_diag_2" type="car" route="route_diagonal_2" begin="0" end="3600" vehsPerHour="100" departLane="best"/>
    <flow id="flow_diag_3" type="car" route="route_diagonal_3" begin="0" end="3600" vehsPerHour="110" departLane="best"/>
    <flow id="flow_diag_4" type="car" route="route_diagonal_4" begin="0" end="3600" vehsPerHour="90" departLane="best"/>
    
    <flow id="flow_bus_WE" type="bus" route="route_WE_1" begin="0" end="3600" vehsPerHour="40" departLane="best"/>
    <flow id="flow_bus_EW" type="bus" route="route_EW_1" begin="0" end="3600" vehsPerHour="40" departLane="best"/>
    <flow id="flow_bus_SN" type="bus" route="route_SN_1" begin="0" end="3600" vehsPerHour="30" departLane="best"/>
    <flow id="flow_bus_NS" type="bus" route="route_NS_1" begin="0" end="3600" vehsPerHour="30" departLane="best"/>
    
    <flow id="flow_truck_WE" type="truck" route="route_WE_0" begin="0" end="3600" vehsPerHour="50" departLane="best"/>
    <flow id="flow_truck_EW" type="truck" route="route_EW_2" begin="0" end="3600" vehsPerHour="50" departLane="best"/>
    
    <vehicle id="ambulance_1" type="emergency" route="route_WE_1" depart="300" color="1,0,0"/>
    <vehicle id="fire_truck_1" type="emergency" route="route_SN_1" depart="900" color="1,0,0"/>
    <vehicle id="police_1" type="emergency" route="route_EW_1" depart="1500" color="0,0,1"/>
</routes>''',

    "grid3x3.sumo.cfg": '''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="grid3x3.net.xml"/>
        <route-files value="grid3x3.rou.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    
    <processing>
        <time-to-teleport value="-1"/>
        <max-depart-delay value="300"/>
        <ignore-route-errors value="true"/>
    </processing>
    
    <routing>
        <device.rerouting.adaptation-steps value="180"/>
    </routing>
    
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
        <duration-log.statistics value="true"/>
    </report>
    
    <gui_only>
        <gui-settings-file value="grid3x3.settings.xml"/>
        <start value="true"/>
    </gui_only>
</configuration>''',

    "grid3x3.settings.xml": '''<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <scheme name="real world">
        <viewport y="250" x="250" zoom="80"/>
        <delay value="100"/>
    </scheme>
</viewsettings>'''
}

# Write all files
print("\n📝 Creating network files...")
for filename, content in files.items():
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  ✓ {filename}")

# Generate the network with LEFT-HAND TRAFFIC
print("\n🔧 Generating LEFT-HAND traffic network (Sri Lankan style)...")
try:
    result = subprocess.run(
        [
            'netconvert',
            '--node-files', os.path.join(output_dir, 'grid3x3.nod.xml'),
            '--edge-files', os.path.join(output_dir, 'grid3x3.edg.xml'),
            '--output-file', os.path.join(output_dir, 'grid3x3.net.xml'),
            '--lefthand'  # THIS IS THE KEY FLAG FOR LEFT-HAND TRAFFIC!
        ],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    
    if result.returncode == 0:
        print("  ✓ grid3x3.net.xml generated with LEFT-HAND traffic!")
    else:
        print(f"  ✗ Error generating network:")
        print(result.stderr)
except FileNotFoundError:
    print("  ✗ netconvert not found in PATH")
    print("\n⚠️  Please run manually:")
    print(f"  cd {output_dir}")
    print("  netconvert --node-files=grid3x3.nod.xml --edge-files=grid3x3.edg.xml --output-file=grid3x3.net.xml --lefthand")

print("\n" + "="*70)
print("✅ LEFT-HAND Traffic Network Complete! (Sri Lankan Style 🇱🇰)")
print("="*70)
print(f"\n📂 Location: {output_dir}")
print("\n🚗 Traffic Direction: LEFT-HAND (like Sri Lanka, UK, India, Japan)")
print("\n🎯 Network: 9 signalized intersections")
print("💡 Vehicles will drive on the LEFT side of the road!")
print("\n🚀 Ready to test your traffic management system!")
print("="*70)