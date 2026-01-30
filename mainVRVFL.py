from VRVFL import VRVFL
import numpy as np
import random
import os
import argparse
import torch
from constants import NUM_COMMUNICATION_ROUNDS, NUM_LOCAL_STEPS

def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Run VRVFL experiment.")
    parser.add_argument('--alpha', type=float, default=0.8, help='Alpha value for VRVFL')
    parser.add_argument('--save_dir_name', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--rounds", type=int, default=NUM_COMMUNICATION_ROUNDS, help="Number of training rounds")
    parser.add_argument("--localsteps", type=int, default=NUM_LOCAL_STEPS, help="Number of local SGD steps")

    args = parser.parse_args()

    # Set random seeds
    set_all_seeds(args.seed)

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir_name, exist_ok=True)

    # Prepare result lists
    accuracy_list = []
    loss_list = []
    activeLearningRate = []

    # Run VRVFL
    VRVFL(accuracy_list, loss_list, args.alpha, args.save_dir_name, args.localsteps , args.verbose, args.rounds)

if __name__ == "__main__":
    main()


