# This file contains functions for calculating path-loss, shadowing, and fading terms. 

from constants import VEHICLE_ANTENNA_HEIGHT, RSU_ANTENNA_HEIGHT, CARRIER_FREQUENCY, SPEED_OF_LIGHT, SHADOWING_STD_DEV, CHANNEL_FEEDBACK_DELAY, VEHICLE_TRANSMIT_POWER, NOISE_POWER_DENSITY, BW_PER_RB, ANTENNA_GAIN, DEFAULT_EPSILON
from math import log10, log2, exp, log
import random
import numpy as np
from scipy.special import j0 as bessel0
# PathLoss

def PathLoss(distance): # WINNER+ B1 LOS - Table4.1
    if distance <= 3:
        distance =3

    dBP = 4 * (VEHICLE_ANTENNA_HEIGHT - 1.0) * (RSU_ANTENNA_HEIGHT - 1.0) * CARRIER_FREQUENCY / SPEED_OF_LIGHT # breakpoint distance
    if distance <= dBP:
        pathLoss = 22.7 * log10(distance) + 27.0 + 20.0 * log10(CARRIER_FREQUENCY / 1e9)
    else:
        pathLoss = 40.0 * log10(distance) + 7.56 - 17.3 * log10((RSU_ANTENNA_HEIGHT - 1.0)) - 17.3 * log10((VEHICLE_ANTENNA_HEIGHT - 1.0)) + 2.7 * log10(CARRIER_FREQUENCY / 1e9)
    
    return pathLoss # dB
    
def Shadowing(): 
    return random.gauss(0, SHADOWING_STD_DEV)

def WirelessProperties(speed, distance):
    PL= PathLoss(distance)
    shadowing= Shadowing()
    largeScaleGain_dB= ANTENNA_GAIN + VEHICLE_TRANSMIT_POWER - PL - shadowing
    largeScaleGain= 10**(largeScaleGain_dB/10)
    noisePower = (10 ** (NOISE_POWER_DENSITY / 10)) * BW_PER_RB
    h_hat_squared = np.random.exponential(1)
    h_tilde_squared = np.random.exponential(1)
    dopplerFreq= speed * CARRIER_FREQUENCY / SPEED_OF_LIGHT
    epsilon= bessel0(2*np.pi*dopplerFreq*CHANNEL_FEEDBACK_DELAY)
    trueCapacity=BW_PER_RB * log2(1 + (largeScaleGain * h_hat_squared * epsilon**2)/(noisePower + largeScaleGain * h_tilde_squared * (1-epsilon**2))) # bps
    return h_hat_squared, h_tilde_squared, epsilon, largeScaleGain, noisePower, trueCapacity

def CapacityUpperLimit(largeScaleGain, h_hat_squared, noisePower, epsilon):
    return BW_PER_RB * log2(1 + ((largeScaleGain * h_hat_squared * epsilon**2) / (largeScaleGain* DEFAULT_EPSILON*(1-epsilon**2)+ noisePower))), exp(noisePower/(largeScaleGain * (1 - epsilon**2)))  # bps, exp(bv/av)=lambda

 # bps

def weighted_sampling_without_replacement(u_vector, N):
    u_vector = np.array(u_vector, dtype=float)
    total_sum = np.sum(u_vector)
    if total_sum == 0:
        raise ValueError("All entries in u_vector are zero.")
    
    # Normalize so that the sum is N
    u_vector = u_vector * (N / total_sum)
    
    selected_indices = []
    
    for _ in range(N):
        # Compute probabilities (weights re-normalized to sum to 1)
        remaining_weights = u_vector.copy()
        remaining_weights[selected_indices] = 0  # Zero out selected entries
        total_remaining = np.sum(remaining_weights)
        
        if total_remaining <= 0:
            raise RuntimeError("Not enough remaining weight to sample without replacement.")
        
        probabilities = remaining_weights / total_remaining
        
        # Sample an index
        chosen_index = np.random.choice(len(u_vector), p=probabilities)
        selected_indices.append(chosen_index)
    
    return selected_indices

def YtoR(Y,vehicle): # get rate from y=exp(-hv^2/av)
    f = -(vehicle.h_hat_squared * vehicle.epsilon**2)/(log(Y)* (1-vehicle.epsilon**2))
    rate = BW_PER_RB * log2(1 + f) # bps
    return rate

def getSuccessfullyParticipatedVehicles(feasibleVehicles, selected_indices):
    successfulParticipants = []
    for idx in selected_indices:
        vehicle = feasibleVehicles[idx]
        vehicleRate= YtoR(vehicle.yt, vehicle)
        if vehicleRate < vehicle.trueCapacity:
            successfulParticipants.append(vehicle)
    return successfulParticipants