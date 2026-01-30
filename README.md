# ICC2026 VRVFL Simulation
Repository for the codes used for simulations in the paper titled "VR-VFL: Joint Rate and Client Selection for Vehicular Federated Learning Under Imperfect CSI" published in IEEE ICC 2026 Conference.

# Preprocessing
Download CIFAR10 Python version from the official site from:
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Run 'PreProcessing.py' script from the same directory as the downloaded dataset.

# Running the Script
VR-VFL simulations can be run with the 'mainVRVFL.py' script. The simulation parameters can be changed in 'constants.py', and the input arguments to 'mainVRVFL.py' script. 

Example terminal command to run the simulation:
python mainVRVFL.py --alpha 0.4 --seed 42 --localsteps 5 --rounds 50 --save_dir_name ./results/
