import os
import sys
import traci

# Check if SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# CHANGE THIS to your actual config file name!
config_file = "scenarios/mapishara.sumo.cfg"

# Check if file exists
if not os.path.exists(config_file):
    print(f"ERROR: Config file not found: {config_file}")
    print("Files in scenarios folder:")
    print(os.listdir("scenarios"))
    sys.exit(1)

print("Starting SUMO simulation...")

# Start SUMO (with GUI so you can see it)
sumo_binary = "sumo-gui"
sumo_cmd = [sumo_binary, "-c", config_file]

traci.start(sumo_cmd)

print("Simulation started! Running for 100 steps...")

# Run simulation for 100 steps
for step in range(100):
    traci.simulationStep()
    
    # Get information every 10 steps
    if step % 10 == 0:
        vehicle_count = traci.vehicle.getIDCount()
        print(f"Step {step}: {vehicle_count} vehicles in simulation")

print("Simulation complete!")
traci.close()