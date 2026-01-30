from Learning import get_class_indices, get_model, get_dataloader_from_tensors, assign_data_to_vehicle, train_local, custom_aggregate_vehicles
import torch
from Physical import Highway
from constants import NOF_SAMPLES_PER_CLASS_PER_VEHICLE, IID_DISTRIBUTION, NUM_CLASSES_PER_VEHICLE_MIN, NUM_CLASSES_PER_VEHICLE_MAX, NUM_COMMUNICATION_ROUNDS, BATCH_SIZE, NUM_LOCAL_STEPS, LOCAL_LEARNING_RATE, NUM_TOTAL_TRAINING_SAMPLES, NOF_SAMPLES_PER_VEHICLE_MIN, NOF_SAMPLES_PER_VEHICLE_MAX, NUM_RESOURCE_BLOCKS, DEFAULT_EPSILON,MODEL_SIZE, BW_PER_RB, MINIMUM_RATE_SUCCESS_CHANCE_THRESHOLD, FED_PROX_TERM, LR_DECAY_ROUND
import random
from BCDalgorithm import solve_bcd_optimization_v2 as bcd_alg
# from BCD_v3 import solve_bcd_optimization_v2 as bcd_alg
import cvxpy as cp
import numpy as np
from Wireless import getSuccessfullyParticipatedVehicles, YtoR, weighted_sampling_without_replacement
from math import exp, log2, log
import sys
from concurrent.futures import ThreadPoolExecutor
import json
import os


def save_training_progress(history, model, current_round, save_dir):
    # 1. Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Save the training history (always overwrite)
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    
    # 3. Save the model every 200 rounds
    if current_round % 200 == 0 and current_round > 0:
        model_path = os.path.join(save_dir, f"global_model_round.pt")
        torch.save(model.state_dict(), model_path)


# NUM_COMMUNICATION_ROUNDS=1
def uniformsampler_method(accuracy_list, loss_list, alpha, save_dir_name = None, num_local_steps = NUM_LOCAL_STEPS, verbose = True, num_comm_rounds = NUM_COMMUNICATION_ROUNDS):
    VRVFLtime = 0
    roundtimes= []
    training_tensors = torch.load("./cifar10_tensors/train_images.pt", weights_only=True)
    labels_tensor = torch.load("./cifar10_tensors/train_labels.pt", weights_only=True)
    test_tensors = torch.load("./cifar10_tensors/test_images.pt", weights_only=True)
    test_labels_tensor = torch.load("./cifar10_tensors/test_labels.pt", weights_only=True)
    
    # --- Initialize Global Model ---
    global_model = get_model("cifar10")
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS for Mac
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA if available
    else:
        device = torch.device("cpu")   # Default to CPU
    global_model.to(device)
    # --- Initialize Highway ---
    myHighway = Highway()
    myHighway.runHighway(3600) # Run for 1 hour to initialize vehicles
    currentVehicles = myHighway.getActiveVehicles()
    if verbose:
        print(f"Number of vehicles: {len(currentVehicles)}")
    
    # --- Initialize Vehicles ---
    class_indices_precomputed = get_class_indices(labels_tensor)
    for vehicle in currentVehicles:
        if IID_DISTRIBUTION == True:
            number_of_classes = 1
            number_of_samples = 1 # will be ignored
        else: 
            number_of_classes = random.randint(NUM_CLASSES_PER_VEHICLE_MIN, NUM_CLASSES_PER_VEHICLE_MAX)
            number_of_samples = random.randint(NOF_SAMPLES_PER_VEHICLE_MIN, NOF_SAMPLES_PER_VEHICLE_MAX)
        assign_data_to_vehicle(labels_tensor, vehicle, NOF_SAMPLES_PER_CLASS_PER_VEHICLE, IID_DISTRIBUTION, class_indices_precomputed, number_of_classes, number_of_samples)


    for round in range(num_comm_rounds):
        if verbose:
            print('=================================')
            print(f"ROUND {round + 1}/{NUM_COMMUNICATION_ROUNDS}")
        currentVehicles = myHighway.getActiveVehicles()
        for vehicle in currentVehicles:
            if not hasattr(vehicle, 'data_indices') or not vehicle.data_indices:
                if IID_DISTRIBUTION == True:
                    number_of_classes = 1
                    number_of_total_samples_current_vehicle = 1
                else: 
                    number_of_classes = random.randint(NUM_CLASSES_PER_VEHICLE_MIN, NUM_CLASSES_PER_VEHICLE_MAX)
                    number_of_total_samples_current_vehicle = random.randint(NOF_SAMPLES_PER_VEHICLE_MIN, NOF_SAMPLES_PER_VEHICLE_MAX)
                assign_data_to_vehicle(labels_tensor, vehicle, NOF_SAMPLES_PER_CLASS_PER_VEHICLE, IID_DISTRIBUTION, class_indices_precomputed, number_of_classes, number_of_total_samples_current_vehicle)

        '''BCD ALGORITHM HERE!!!'''
        feasibleVehicles = [] # V_t^feas - vehicles with Rmin < Rmax
        
        Dv_overD_vector= [] # D_v/D for each vehicle
        bcdLambda_vector = []
        y_min_vec= []
        y_max_vec= []
        D_total_feasible = 0 # D_total_fes = sum(D_v) for all vehicles in V_t^feas 

        for vehicle in currentVehicles:
            if vehicle.Rmin < vehicle.Rmax:
                f = 2**(vehicle.Rmin / BW_PER_RB) - 1
                a = (1 - vehicle.epsilon**2) /vehicle.epsilon**2
                a_v = f * a
                b = vehicle.noisePower / (vehicle.largeScaleGain * vehicle.epsilon**2)
                b_v = f * b
                minimumRateSuccessChance = 1-(exp(-(vehicle.h_hat_squared-b_v)/(a_v)))
                if minimumRateSuccessChance> MINIMUM_RATE_SUCCESS_CHANCE_THRESHOLD: # remaining vehicles are in an impossible condition to transmit.
                    feasibleVehicles.append(vehicle)
                    d = -log(1 - MINIMUM_RATE_SUCCESS_CHANCE_THRESHOLD)
                    fmax = (vehicle.h_hat_squared)/(a * d + b)
                    vehicle.ymax = exp(- (vehicle.h_hat_squared * vehicle.epsilon**2 ) /((1 - vehicle.epsilon**2) * fmax ))

        for vehicle in feasibleVehicles:
            D_total_feasible+= len(vehicle.data_indices)

        for vehicle in feasibleVehicles:
            vehicle.D_ratio = len(vehicle.data_indices) / D_total_feasible # D_v/D for each vehicle
            Dv_overD_vector.append(vehicle.D_ratio) # D_v/D for each vehicle 
            bcdLambda_vector.append(vehicle.bcdLambda) # 1/(1-LambdaY) for each vehicle
            # vehicle.calculateVehicle_ymin_ymax() # only calculate these if the vehicle is feasible, otherwise ymin causes overflow.
            y_min_vec.append(vehicle.ymin) # y_min for each vehicle
            y_max_vec.append(vehicle.ymax) # y_max for each vehicle

        if verbose:
            print(f"Number of feasible vehicles:{len(feasibleVehicles)}")
        
        Dv_overD_vector = np.array(Dv_overD_vector)  # Convert list to NumPy array
        bcdLambda_vector = np.array(bcdLambda_vector)  # Also convert these if needed
        y_max_vec = np.array(y_max_vec)
        y_min_vec = np.array(y_min_vec)
        # Ensure entries in y_min_vec are not smaller than DEFAULT_EPSILON
        y_min_vec = np.maximum(y_min_vec, DEFAULT_EPSILON)
        results = bcd_alg(len(feasibleVehicles), NUM_RESOURCE_BLOCKS, alpha,Dv_overD_vector, bcdLambda_vector, 1.0, DEFAULT_EPSILON, y_max_vec, y_min_vec, 50, 1e-3, None, None, cp.SCS, False, DEFAULT_EPSILON)

        u_t = results['u'] # u_t is the vector of size |V_t^feas|
        y_vec = results['y'] # y_vec is the vector of size |V_t^feas|

        # Assign the solved u and y values to the corresponding vehicles
        for idx, vehicle in enumerate(feasibleVehicles):
            vehicle.u_vt =1  # Assign the solved u value
            vehicle.yt = y_vec[idx]  # Assign the solved y value

        selected_indices = random.sample(range(len(feasibleVehicles)), NUM_RESOURCE_BLOCKS)
        successfulParticipants=getSuccessfullyParticipatedVehicles(feasibleVehicles, selected_indices) # get the vehicles that successfully participated in the communication round
        if verbose:
            print(f"Number of successful participants: {len(successfulParticipants)}")

        minSuccessfulRate=float('inf')
        for vehicle in successfulParticipants:
            currentVehicleRate=YtoR(vehicle.yt,vehicle)
            f = 2**(currentVehicleRate / BW_PER_RB) - 1
            a_v = f * (1 - vehicle.epsilon**2) /vehicle.epsilon**2
            b_v = f * vehicle.noisePower / (vehicle.largeScaleGain * vehicle.epsilon**2)
            vehicle.P_A = 1 - exp(-(vehicle.h_hat_squared-b_v)/(a_v))
            if currentVehicleRate < minSuccessfulRate:
                minSuccessfulRate = currentVehicleRate
                roundTime = MODEL_SIZE / currentVehicleRate
        

        learning_rate = LOCAL_LEARNING_RATE / (1 + round // LR_DECAY_ROUND)  # Decay learning rate
        if verbose:
            print(f"roundTime: {roundTime:.2f} seconds")


        for vehicle in successfulParticipants:
            vehicleDataLoader = get_dataloader_from_tensors(training_tensors, labels_tensor, vehicle.data_indices, BATCH_SIZE)
            vehicle.model_state_dict = train_local(vehicleDataLoader, num_local_steps, learning_rate, device, global_model.state_dict(), "cifar10",FED_PROX_TERM)

        
        '''Aggregate the models'''
        aggregated_state_dict = custom_aggregate_vehicles(successfulParticipants, epsilon=1e-9)

        global_model.load_state_dict(aggregated_state_dict)

        # Evaluate the global model
        global_model.eval()
        test_dataloader = get_dataloader_from_tensors(test_tensors, test_labels_tensor, list(range(len(test_labels_tensor))), BATCH_SIZE)
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()  # Define the loss function
        total_loss = 0.0  # Initialize total loss

        with torch.no_grad():
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                loss = criterion(outputs, labels)  # Compute the loss
                total_loss += loss.item()  # Accumulate the loss
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        average_loss = total_loss / len(test_dataloader)  # Compute average loss
        if verbose:
            print(f"Round {round + 1}: Average Loss = {average_loss:.4f}")

        accuracy = correct / total
        if verbose:
            print(f"Round {round + 1}: Global Model Accuracy = {accuracy:.4f}")
        
        accuracy_list.append(accuracy)
        loss_list.append(average_loss)
        myHighway.runHighway(roundTime)
        roundtimes.append(roundTime)
        VRVFLtime += roundTime
        history = {
            "accuracy": accuracy_list,
            "loss": loss_list,
            "TrainingTime": VRVFLtime,
            "roundTimes": roundtimes,
            "alpha": alpha,
            "learning_rate": LOCAL_LEARNING_RATE,
            "fedprox_term": FED_PROX_TERM,
            "iid_distribtuion": IID_DISTRIBUTION,
            "number of local SGD steps": num_local_steps,
            "lrdecayparameter":LR_DECAY_ROUND

        }
        save_training_progress(history, global_model, round, save_dir_name)
