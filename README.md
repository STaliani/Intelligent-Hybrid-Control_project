# Intelligent-Hybrid_Control

This repository contains a simplified implementation of Erik Berger et al 2014 .
It aims to employ Dynamic Mode Decomposition to detect and estimate external disturbaces acting on a robot during a task.
Here you can find the application on a 2R planar robot subject to gravity.

This project was realized to complete the exam of Intelligent & Hybrid control, MS Control Engineering, Sapienza University of Rome, ay 2022/2023

To execute the code follow the following istructions 

- First install the repository
```
git clone https://github.com/STaliani/Intelligent-Hybrid_Control.git
```

- Then create and activate the environment
```
cd Intelligent-Hybrid_Control
conda config --add channels conda-forge # if you don't have the conda-forge channel already
mamba env create -f environment.yml
conda activate ihc_project
pip install .
```
- Last run the python script
```
python run/run.py
```
