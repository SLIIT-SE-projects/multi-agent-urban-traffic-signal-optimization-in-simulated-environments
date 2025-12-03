import os
import random
import sys
import subprocess
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "simulation"))

NET_DIR = os.path.join(SIM_DIR, "networks")
ROUTE_DIR = os.path.join(SIM_DIR, "routes")
CONFIG_DIR = os.path.join(SIM_DIR, "config")

# Ensure folders exist
for folder in [NET_DIR, ROUTE_DIR, CONFIG_DIR]:
    os.makedirs(folder, exist_ok=True)


class ScenarioGenerator:
    def __init__(self, filename_base="test_scenario", seed=None):
        self.filename_base = filename_base
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self.net_file = os.path.join(NET_DIR, f"{self.filename_base}.net.xml")
        self.rou_file = os.path.join(ROUTE_DIR, f"{self.filename_base}.rou.xml")
        self.cfg_file = os.path.join(CONFIG_DIR, f"{self.filename_base}.sumocfg")

    # ---------- NETWORK ----------
    def generate_network(self):
        """
        Uses SUMO's netgenerate tool to create a 3x3 grid with TLS.
        Uses subprocess list args to avoid Windows quoting issues.
        """
        print("Generating Grid Network...")

        cmd = [
            "netgenerate",
            "--grid",
            "--grid.number=3",
            "--grid.length=200",
            "--tls.guess",
            "--tls.guess.threshold", "0",
            "--tls.default-type", "actuated",
            "--output-file", self.net_file
        ]

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            raise RuntimeError(
                "netgenerate was not found. Make sure SUMO 'bin' folder is on PATH."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"netgenerate failed with error: {e}")

        if not os.path.exists(self.net_file):
            raise RuntimeError("Network generation failed: net file was not created.")

        print("Success.")

    # ---------- EDGE HELPERS ----------
    def get_valid_edges(self):
        """
        Reads the generated .net.xml and returns valid (non-internal) edge IDs.
        """
        tree = ET.parse(self.net_file)
        edges = []
        for e in tree.findall(".//edge"):
            edge_id = e.get("id")
            func = e.get("function")
            # Skip internal edges (like ':junction_0')
            if edge_id and func != "internal" and not edge_id.startswith(":"):
                edges.append(edge_id)

        if len(edges) < 2:
            raise RuntimeError("Not enough valid edges found in net file.")

        return edges

    def pick_far_edges(self, edges):
        """
        Heuristic to pick 'far' start/end edges without running routing tools.
        We just pick two random distinct edges;
        """
        return random.sample(edges, 2)

    # ---------- ROUTES ----------
    def generate_routes(self):
        print("Generating Trips (will convert with duarouter)...")

        edges = self.get_valid_edges()

        trips_file = os.path.join(ROUTE_DIR, f"{self.filename_base}.trips.xml")

        with open(trips_file, "w") as f:
            f.write("""<routes>
        <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5"
            maxSpeed="15" guiShape="passenger"/>
        <vType id="ambulance" vClass="emergency" accel="4.0" decel="6.0" sigma="0.2"
            length="6" minGap="2.5" maxSpeed="30" guiShape="emergency" speedFactor="1.5">
            <param key="has.bluelight.device" value="true"/>
        </vType>
    """)

            # background trips
            num_bg = 50
            for i in range(num_bg):
                depart = i * 2
                frm, to = random.sample(edges, 2)
                f.write(f'    <trip id="c_{i}" type="car" depart="{depart}" from="{frm}" to="{to}" />\n')

            # EV trip
            ev_from, ev_to = random.sample(edges, 2)
            f.write(f'    <trip id="EV_1" type="ambulance" depart="40" from="{ev_from}" to="{ev_to}" />\n')
            f.write("</routes>")

        # convert trips -> routes
        print("Running duarouter to build actual routes...")
        cmd = [
            "duarouter",
            "-n", self.net_file,
            "-r", trips_file,
            "-o", self.rou_file,
            "--ignore-errors",
            "--repair"
        ]
        subprocess.run(cmd, check=True)

        if not os.path.exists(self.rou_file):
            raise RuntimeError("duarouter failed: route file not created.")

        print("Routes generated:", self.rou_file)


    # ---------- CONFIG ----------
    def generate_config(self):
        """
        Generates .sumocfg linking net and routes.
        Paths stored relative to CONFIG_DIR so SUMO can resolve via '..'.
        """
        print("Generating Config File...")

        net_rel = os.path.relpath(self.net_file, CONFIG_DIR)
        rou_rel = os.path.relpath(self.rou_file, CONFIG_DIR)

        with open(self.cfg_file, "w") as f:
            f.write(f"""<configuration>
    <input>
        <net-file value="{net_rel}"/>
        <route-files value="{rou_rel}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
    </time>
    <gui_only>
        <start value="true"/>
    </gui_only>
</configuration>""")

    # ---------- RUN ----------
    def run(self):
        self.generate_network()
        self.generate_routes()
        self.generate_config()

        print("\nSUCCESS: Test scenario created!")
        print("NET  :", self.net_file)
        print("ROUTE:", self.rou_file)
        print("CFG  :", self.cfg_file)


if __name__ == "__main__":
    # Optional args:
    #   python scenario_generator.py [base_name] [seed]
    base_name = "test_scenario"
    seed = None

    if len(sys.argv) > 1:
        base_name = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            seed = int(sys.argv[2])
        except ValueError:
            seed = None

    generator = ScenarioGenerator(filename_base=base_name, seed=seed)
    generator.run()
