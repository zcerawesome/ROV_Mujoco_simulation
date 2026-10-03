import os

from simulation import MuJoCoBase

xml_path = "models/rov.xml"

# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)
xml_path = abspath

print("Creating simulation environment")
base = MuJoCoBase(xml_path, controller_host="localhost")

print("Connected. Running headless, driven by controller.py's movement commands...")
base.run_networked()
base.close()