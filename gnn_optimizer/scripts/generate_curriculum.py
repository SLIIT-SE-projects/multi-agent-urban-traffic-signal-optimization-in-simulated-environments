import xml.etree.ElementTree as ET
import random
import os

# CONFIGURATION
SOURCE_FILE = "simulation/routes.rou.xml" # Your current file
OUTPUT_DIR = "simulation"

def filter_routes(source_path, output_path, keep_probability):
    """
    Reads the route file and keeps vehicles based on a probability.
    """
    tree = ET.parse(source_path)
    root = tree.getroot()
    
    # We want to keep all route definitions, vTypes, etc.
    # We only filter 'vehicle' tags.
    vehicles_to_remove = []
    
    for child in root:
        if child.tag == 'vehicle':
            # Emergency vehicles should always be kept
            if 'type' in child.attrib and child.attrib['type'] == 'emergency':
                continue
            
            # Randomly drop vehicles
            if random.random() > keep_probability:
                vehicles_to_remove.append(child)
    
    # Remove the selected vehicles
    for veh in vehicles_to_remove:
        root.remove(veh)
        
    # Save
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)
    print(f"✅ Generated {output_path} with ~{int(keep_probability*100)}% traffic.")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Error: Could not find {SOURCE_FILE}")
    else:
        # Create Levels
        filter_routes(SOURCE_FILE, f"{OUTPUT_DIR}/routes_easy.xml", 0.3)   # 30% Traffic
        filter_routes(SOURCE_FILE, f"{OUTPUT_DIR}/routes_medium.xml", 0.6) # 60% Traffic
        filter_routes(SOURCE_FILE, f"{OUTPUT_DIR}/routes_hard.xml", 1.0)   # 100% Traffic (Copy)