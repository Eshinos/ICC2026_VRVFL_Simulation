import math
import random
from Wireless import WirelessProperties, CapacityUpperLimit
from constants import LANE_LENGTH, LANE_WIDTH, NUM_LANES, MIN_VEHICLE_SPEED, MAX_VEHICLE_SPEED, VEHICLE_ARRIVAL_RATE, RSUpositionsX, RSUpositionY, MODEL_SIZE, BW_PER_RB
from math import exp, log

class Vehicle:
    def __init__(self, speed, positionX, positionY):
        self.speed = speed # m/s
        self.positionX = positionX # meters from the start of the lane to the current position
        self.positionY = positionY
        self.remainingSojournTime = LANE_LENGTH / speed # seconds
        self.h_hat_squared = -99
        self.h_tilde_squared = -99
        self.epsilon = -99
        self.largeScaleGain = -99
        self.noisePower = -99
        self.trueCapacity = -99 # bps

        # Following variables are for, BCD solver to use in the main loop.
        self.Rmin= -99
        self.Rmax = -99
        self.a_vt = -99
        self.b_vt = -99
        self.ymin = -99
        self.ymax = -99
        self.bcdLambda = -99 # 1/(1-LambdaY) is the first objective
        #will be assigned on the main loop
        # self.data_indices = []
        # self.D_ratio = -99
        # self.model_state_dict
        # self.u_vt
        # self.P_A 
        # self.yt
        # self.Rate

    def updateWirelessProperties(self):
        distance = math.sqrt(min(abs(x - self.positionX) for x in RSUpositionsX)**2 + (self.positionY - RSUpositionY)**2)
        self.h_hat_squared, self.h_tilde_squared, self.epsilon, self.largeScaleGain, self.noisePower, self.trueCapacity = WirelessProperties(self.speed, distance)
        self.Rmin = MODEL_SIZE / self.remainingSojournTime
        self.Rmax, self.bcdLambda = CapacityUpperLimit(self.largeScaleGain, self.h_hat_squared, self.noisePower, self.epsilon)
        if self.Rmax > self.Rmin:
            exponent = - (self.h_hat_squared * self.epsilon**2 ) / ((1 - self.epsilon**2) * (2**(self.Rmin / BW_PER_RB)-1))
            self.ymin = exp(exponent) # to avoid overflow
        # self.ymin = exp(- (self.h_hat_squared * self.epsilon**2 ) /((1 - self.epsilon**2) * 2 ** (self.Rmin / BW_PER_RB) )) 
        self.ymax = exp(- (self.h_hat_squared * self.epsilon**2 ) /((1 - self.epsilon**2) * (2 ** (self.Rmax / BW_PER_RB)-1) ))



class TrafficLane:
    def __init__(self, positionY):
        self.vehicles = []
        self.nextArrivalTime = self.computeNextArrivalTime() # seconds
        self.positionY = positionY
        
    def addVehicle(self):
        vehicle=Vehicle(speed=random.uniform(MIN_VEHICLE_SPEED, MAX_VEHICLE_SPEED), positionX=0, positionY=self.positionY)
        self.vehicles.append(vehicle)
        return vehicle

    def computeNextArrivalTime(self):
        # poisson process inter arrival time
        U = random.uniform(0, 1)  
        return -math.log(U) / VEHICLE_ARRIVAL_RATE 
    
    def moveVehicle(self, vehicle, travelTime, toRemove):
        if vehicle.remainingSojournTime <= travelTime:
                toRemove.append(vehicle)
        else:
            vehicle.positionX += vehicle.speed * travelTime
            vehicle.remainingSojournTime -= travelTime
            vehicle.updateWirelessProperties()

    def runLane(self, forwardTime):
        toRemove = []
        for vehicle in self.vehicles:
            self.moveVehicle(vehicle, forwardTime, toRemove) 
        
        remainingTime = forwardTime
        while remainingTime > self.nextArrivalTime: # add vehicles until the next arrival time is greater than the remaining time
            vehicle = self.addVehicle()
            remainingTime -= self.nextArrivalTime
            self.moveVehicle(vehicle, remainingTime, toRemove)
            self.nextArrivalTime = self.computeNextArrivalTime()

        self.nextArrivalTime -= remainingTime
        if toRemove:
            for vehicle in toRemove:
                self.vehicles.remove(vehicle)
        
class Highway:
    def __init__(self):
        self.lanes = [TrafficLane(i*LANE_WIDTH + LANE_WIDTH/2) for i in range(NUM_LANES)]

    def runHighway(self, forwardTime):
        for lane in self.lanes:
            lane.runLane(forwardTime)

    def getActiveVehicles(self):
        active_vehicles = []
        for lane in self.lanes:
            active_vehicles.extend(lane.vehicles)
        return active_vehicles
