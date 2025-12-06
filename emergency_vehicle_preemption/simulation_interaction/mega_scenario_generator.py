import os
import random
import subprocess
import sumolib

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.abspath(os.path.join(BASE_DIR, "../simulation"))
NET_DIR = os.path.join(SIM_DIR, "networks")
ROUTE_DIR = os.path.join(SIM_DIR, "routes")
CONFIG_DIR = os.path.join(SIM_DIR, "config")

for folder in [NET_DIR, ROUTE_DIR, CONFIG_DIR]:
    os.makedirs(folder, exist_ok=True)

class MegaScenarioGenerator:
    def __init__(self, filename_base="mega_scenario"):
        self.filename_base = filename_base
        self.net_file = os.path.join(NET_DIR, f"{self.filename_base}.net.xml")
        self.trips_file = os.path.join(ROUTE_DIR, f"{self.filename_base}.trips.xml")
        self.rou_file = os.path.join(ROUTE_DIR, f"{self.filename_base}.rou.xml")
        self.cfg_file = os.path.join(CONFIG_DIR, f"{self.filename_base}.sumocfg")

    def generate_network(self):
        """Generates a 4x4 Grid Network with Actuated Lights."""
        print("1. Generating 4x4 Grid Network...")
        cmd = [
            "netgenerate",
            "--grid", "--grid.number=4", "--grid.length=200",
            "--tls.guess", "--tls.guess.threshold", "0",
            "--tls.default-type", "actuated",
            "--output-file", self.net_file
        ]
        subprocess.run(cmd, check=True)

    def generate_routes(self):
        """Generates 50 EVs and continuous background traffic."""
        print("2. Reading Network to find valid edges...")
        net = sumolib.net.readNet(self.net_file)
        valid_edges = [e.getID() for e in net.getEdges() if e.allows("passenger")]
        
        all_trips = []
        SIMULATION_DURATION = 7200  # 2 Hours
        NUM_EVS = 200               # 200 EVs

        print(f"3. Generating {NUM_EVS} EVs and Background Traffic...")

        # A. Background Traffic (Flows)
        # We generate random flows to keep the grid busy throughout the hour
        for i in range(100):
            begin = random.randint(0, SIMULATION_DURATION - 200)
            src, dst = random.sample(valid_edges, 2)
            all_trips.append({
                "tag": "flow",
                "id": f"flow_{i}",
                "type": "car",
                "begin": begin,
                "end": begin + 600, # Flow lasts 10 mins
                "number": random.randint(5, 15), # 5-15 cars per flow
                "from": src, "to": dst
            })

        # B. 50 Emergency Vehicles (EVs)
        # Spawn one every ~60-70 seconds
        for i in range(NUM_EVS):
            depart = 50 + (i * 70) # Spaced out so they don't overlap too much
            
            # Pick Random Route for THIS EV
            src, dst = random.sample(valid_edges, 2)
            
            all_trips.append({
                "tag": "trip",
                "id": f"EV_{i}", # IDs: EV_0, EV_1, ... EV_49
                "type": "ambulance",
                "depart": depart,
                "from": src, "to": dst
            })

        # SORT BY TIME (Crucial for SUMO)
        # Flows use 'begin', Trips use 'depart'. We normalize to 'time' for sorting.
        all_trips.sort(key=lambda x: x.get("depart", x.get("begin")))

        # WRITE TO FILE
        with open(self.trips_file, "w") as f:
            f.write("""<routes>
    <vType id="car" accel="2.6" decel="4.5" length="5" maxSpeed="15" guiShape="passenger"/>
    <vType id="ambulance" vClass="emergency" accel="4.5" decel="6.0" length="6" maxSpeed="30" guiShape="emergency" speedFactor="1.5">
        <param key="has.bluelight.device" value="true"/>
    </vType>
""")
            for t in all_trips:
                if t["tag"] == "trip":
                    f.write(f'    <trip id="{t["id"]}" type="{t["type"]}" depart="{t["depart"]}" from="{t["from"]}" to="{t["to"]}" />\n')
                else:
                    f.write(f'    <flow id="{t["id"]}" type="{t["type"]}" begin="{t["begin"]}" end="{t["end"]}" number="{t["number"]}" from="{t["from"]}" to="{t["to"]}" />\n')
            
            f.write("</routes>")

        print("4. Running duarouter...")
        subprocess.run(
            ["duarouter", "-n", self.net_file, "-r", self.trips_file, "-o", self.rou_file, "--ignore-errors", "--repair"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def generate_config(self):
        print("5. Generating Config...")
        net_rel = os.path.relpath(self.net_file, CONFIG_DIR)
        rou_rel = os.path.relpath(self.rou_file, CONFIG_DIR)
        with open(self.cfg_file, "w") as f:
            f.write(f"""<configuration>
    <input><net-file value="{net_rel}"/><route-files value="{rou_rel}"/></input>
    <time><begin value="0"/><end value="4000"/></time>
</configuration>""")

    def run(self):
        self.generate_network()
        self.generate_routes()
        self.generate_config()
        print(f"\nSUCCESS! Mega scenario created at:\n{self.cfg_file}")

if __name__ == "__main__":
    MegaScenarioGenerator().run()