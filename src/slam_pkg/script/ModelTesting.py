import GPy
from zipfile import ZipFile


with ZipFile("/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel/GP_Model.json.zip", 'r') as zObject: 
    zObject.extractall(path="/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel")


# myModel = GPy.load('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel/GP_Model.json')