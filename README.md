# Spezialisierung-1

This repository contains the project **Spezialisierung-1**, developed by [CoCoKayya18](https://github.com/CoCoKayya18). The project is primarily written in Python, with additional components in CMake and C++.

## Project Structure

The repository is organized as follows:

- `src/`: Directory for source code files.
  - `slam_pkg/`:
      - `Scaler/`: Folder containing all the Scalers created, naming is consitent with the model name
      - `data/`: Contains all the data in csv format, and also plots of theses data 
      - `launch/`: Contains launch file for rosbag data gathering
      - `myMLmodel/`: Contains all trained ML models, with corresponding plots
      - `rosbag_files/`: Contains all recorded rosbag files
      - `script/`: Contains all python scripts for training and testing the models
      - `urdf/`: Contains modified turtlebot burger urdf, which contains the Ground Truth Plugin
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Languages Used

- **Python**: 94.0%
- **CMake**: 4.4%
- **C++**: 1.6%

## Getting Started

To get started with this project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/CoCoKayya18/Spezialisierung-1.git
   ```

2. **Navigate to the project directory**:
  ```bash
  cd Spezialisierung-1
  ```

3. Build the project
  ```bash
  catkin_make
  ```







