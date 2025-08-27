# Sionna for ISAC

This repository contains the source code for the Integrated Sensing and Communication (ISAC) Simulation using Sionna. Below is a detailed description of the files and their contents:

## 1. utils: Utility Python Files

- **`channel.py`**: Contains the `ChannelComputation` class, which manages and computes channel characteristics, applies noise, and handles saving and loading of channel data.
- **`paths.py`**: Contains the `PathComputation` class, which computes and processes path information for a given scene, including Doppler shifts and delays, and applies thresholds to path data.
- **`rg.py`**: Contains the `ResourceGridSetup` class, which configures and provides access to a resource grid for OFDM systems, including its subcarrier frequencies.
- **`scenariosetup.py`**: Contains the `ScenarioSetup` class, which sets up a vehicular simulation scenario by generating vehicle data, configuring a simulation scene, and managing vehicle positions and velocities.
- **`xml.py`**: Contains the `XMLModifier` class, which modifies XML files to add new elements based on a template and provides methods for proper XML formatting and saving.

## 2. scene: Scene Files

- **`scene.xml`**: The original single-car template scene used as a reference for adding other cars.
- **`mod_scene.xml`**: A modified scenario with the required number of cars, based on the original reference.
- **`mod_scene.csv`**: Contains information about the positions and velocity vectors of cars.
- **`dop_gt.npy`**: Contains ground truth data for Doppler shifts and delays in the scenario.
- **`chl_gt.npy`**: Contains the ground truth OFDM channel data for the scenario.
- **`meshes/`**: Folder containing mesh files (PLY files) for various objects like buildings, floor, and cars.

## 3. analysis.ipynb

A Jupyter Notebook containing code used to analyze the effect of different types of paths and mobility on the channel.

## 4. og.ipynb

A Jupyter Notebook containing nearly all the core code initially used to make the base version of the simulation work.

## 5. modifyxml.py

A Python script that generates a new modified scenario with the required number of cars when executed.

## 6. scenario_gt.py

A Python script to initialize the scene, position the objects in the scenario, perform ground truth analysis to find the true channel, Doppler shift, and delay, and save them as NumPy files.

## 7. values.py

A global variables file used to set all parameters in the simulation. Changes in this file affect parameters for all the scripts.

## Resources

Here are some useful resources for the Integrated Sensing and Communication (ISAC) Simulation using Sionna:

1. **Information on Scattering Coefficients of Materials**:
   - [Absorption Coefficient Chart](https://www.acoustic-supplies.com/absorption-coefficient-chart/)

2. **Sionna Tutorials**:
   - [Sionna Tutorials](https://nvlabs.github.io/sionna/tutorials.html)

3. **Discussions on Sionna GitHub**:
   - [Discussion 450](https://github.com/NVlabs/sionna/discussions/450)
   - [Discussion 475](https://github.com/NVlabs/sionna/discussions/475)
   - [Discussion 516](https://github.com/NVlabs/sionna/discussions/516)
   - [Discussion 523](https://github.com/NVlabs/sionna/discussions/523)

4. **Frequency Dependence of Scattering Coefficients**:
   - R. Luebbers, D. Steich and K. Kunz, "FDTD calculation of scattering from frequency-dependent materials," in IEEE Transactions on Antennas and Propagation, vol. 41, no. 9, pp. 1249-1257, Sept. 1993, doi: 10.1109/8.247751.
   - Keywords: Finite difference methods, Time domain analysis, Scattering, Frequency, Magnetic materials, Convolution, Magnetic confinement, Plasma confinement, Permittivity, Anisotropic magnetoresistance

5. **Weekly Updates Presentations (combined)**:
   - [Google Slides Presentation](https://docs.google.com/presentation/d/1ExG5S0luL7W-TqLdqEjusiJEiLuzWcYruyqAF-UmW-k/edit?usp=sharing)