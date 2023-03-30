import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.optimize import curve_fit
import pandas as pd
import openpyxl

import sys
sys.path.insert(1, '../../../mri_sim')
sys.path.insert(1, '../../..')

sys.path.insert(0, '../../../mri_sim/SSFP')
from simulator import *

##------------------------------------- Functions Section -------------------------------------------
###Calculate the phase shift for current TR (quadratic spoiling)
def QuadraticPhase(N, n):
    phi_quad = 360/N
    phase = phi_quad*n*n/2
    phase = phase%360
 
    #in degree
    return phase

### Quadratic RF spoiling bSSFP - data simulation function
### with Noise and Field inhomogeneous
#Simulate Data with different field inhomogeneous effect (different T2*) and random noise 
def quadratic_RFspoiling_bSSFP (T1, T2, off_res_max, TR, TE, tip_angle, Period, M0, TRnum, M_transverse, off_resonance_f, T2p, testnum, AddNoise, Noise):
    #Array with (T2*num, noise num, off-res f num)
    SimulatedData = np.asarray(np.zeros((1,testnum,off_resonance_f.shape[0])), dtype = float) 
    
    T2Star_GT = np.asarray([], dtype = float) #Store the Ground Truth T2*
   
    
    #Add field inhomogeneous 
    for T2_inhom in T2p:
        SimulatedData_slice = np.asarray(np.zeros((1,off_resonance_f.shape[0])), dtype = float)
        #Store Ground truth T2*------------------------------------
        T2Star_GT = np.append(T2Star_GT, 1/((1/T2)+(1/T2_inhom)))
        
        #Add inhomogeneous, get the off-resonance profile with inhomogeneous effect added
        F_magnitude_inhomo, F_state = AddfieldInhomogeneous(M_transverse, TR, TE, T2_inhom, off_resonance_f)
        
        #Add Noise
        for test in range (0, testnum ):
            if (AddNoise == True):
                F_magnitude_inhomo_noise = F_magnitude_inhomo + Noise[test,:]
            else:
                F_magnitude_inhomo_noise = F_magnitude_inhomo
            SimulatedData_slice = np.append(SimulatedData_slice, F_magnitude_inhomo_noise.reshape(1,off_resonance_f.shape[0]), axis = 0)
        SimulatedData_slice = np.delete(SimulatedData_slice,0,0)
        SimulatedData = np.append(SimulatedData, SimulatedData_slice.reshape(1,testnum,off_resonance_f.shape[0]), axis = 0)
    SimulatedData = np.delete(SimulatedData,0,0)
        
    return SimulatedData, T2Star_GT

###Off-resonance profile magnitude --> F-state magnitude 
def Fstate(M_transverse):
    F_Magnitude = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(M_transverse), axis = 0))/M_transverse.shape[0]

    F_state = np.fft.fftfreq(M_transverse.shape[0], d = 1.0)
    F_state = np.fft.fftshift(F_state)*M_transverse.shape[0]
    
    return np.transpose(F_Magnitude), np.transpose(F_state)

###Add field inhomogeneous to model T2* effect (for F_number F-states, F_number = 6 --> 0,1,2...5 F-states
def AddfieldInhomogeneous(M_transverse, TR, TE, T2prime, off_resonance_f):
   
    #F-state
    F_state_magnitude, F_state = Fstate(M_transverse)
    F_state = np.round(F_state).astype(int) #round F-state all to integer 
    F_state_magnitude_inhomo = F_state_magnitude
    
    index = 0
    for i in F_state:
        #remove the empty state caused by taking 2 period off-res profile 
        if ((i%2) == 0):
            F = np.abs(-i//2) #the right F-state (all cast to positive)
            time = TE+ F *TR
            if (T2prime == 0):
                F_state_magnitude_inhomo[0,index] = F_state_magnitude[0,index]
            else:
                F_state_magnitude_inhomo[0,index] = F_state_magnitude[0,index] *np.exp(-time/T2prime)
        index = index + 1 
    #End
    
    return F_state_magnitude_inhomo, F_state

###T2* Fit 
###Performing T2* Fit using the previous simulated data (F-state)
def T2StarFit(Data, T2Star_GT, Period, TR):
    #testnumber*(T2'number) matrix storing T2* ground truth and measured --> used to calculate error later 
    #each row store measured T2* value for a ground truth T2*
    TE = TR/2
    
    testnumber = np.shape(Data)[1]
    T2s_m_all = np.asarray([], dtype = float)
    #testnumber*2 matrix storing mean error and sd
    Mean = np.asarray([], dtype = float)
    Errorstd = np.asarray([], dtype = float)
    Bias = np.asarray([], dtype = float)

    slicenum = 0 #slice from the dataset, each slice contain several simulated data for one T2*
    
    for GroundTruth in T2Star_GT:
        T2s_m = np.asarray([], dtype = float)
        for test in range (0,testnumber):
            #select each line from dataset 
            F_state_magnitude = Data[slicenum, test, :]
            F_state = np.fft.fftshift(np.fft.fftfreq(F_state_magnitude.shape[-1], d = 1.0))*F_state_magnitude.shape[-1]

            #______________________T2* Fit______________________________________
            

            #fit using 6 or less F-states
            F_state = np.round(F_state).astype(int) #round F-state all to integer 
            F = np.asarray([], dtype = float)
            F_Magnitude = np.asarray([], dtype = float)
            for i in range (0,Period):
                F_index = np.where(F_state == float(2+2*i))

                F = np.append(F, i)
                F_Magnitude = np.append(F_Magnitude, np.abs(F_state_magnitude[F_index]))
            #End

            ## time = TE+F*TR
            time = TE*np.ones(np.shape(F))+ F *TR

            def func(t, A, R2Star):
                return A * np.exp(-R2Star * t)
            #fitted coefficient stored in popt:[A R2*] --> T2* = 1/R2*
            popt, pcov = curve_fit(func, time, F_Magnitude)

            # plt.figure(1)
            # plt.plot(time, F_Magnitude, marker = 'o')
            T2Star_measured = 1/popt[1]
            T2s_m = np.append(T2s_m, T2Star_measured)
        
            #____________________________________________________________
        #For End
        
        #calculate error
        T2s_m_mean = np.mean(T2s_m)
        Mean = np.append(Mean, T2s_m_mean)
        Errorstd = np.append(Errorstd, np.std(T2s_m))
        Bias = np.append(Bias, (T2s_m_mean-GroundTruth)/GroundTruth)
        slicenum = slicenum + 1
    #For End
    return Mean, Errorstd, Bias



##-------------------------------------Simulation Parameters Section-------------------------------------------

#Map resoluation
TR_n = 100

#number of simulation run
testnumber = 1000
AddNoise = True

# the parameters used in this simulation 
#time unit --- second
proton_density = 1.0
T1 = 900e-3
T2 = 44e-3
off_res_samplesize = 1000

TR = np.asarray([], dtype = float)
TR_min = 2e-3
TR_max = 10e-3
TR_increment = (TR_max-TR_min)/TR_n
for i in range (0, TR_n):
    TR = np.append(TR, TR_min+TR_increment*i)

TE = TR/2
#number of TR
Nr = 500
# tip angle alpha 10
tip_angle = 10
alpha = [np.deg2rad(tip_angle)] * Nr

#initial magnetisation 
M0 = 1
#off resonance profile range
f = 4
off_res = np.linspace(-np.pi*f, np.pi*f, off_res_samplesize+1)
off_res = off_res[0:-1]
dphi = [0]* Nr
#T2* range 13 ms to 53 ms
T2s_min = 13
T2s_max = 53
T2p = np.asarray([], dtype = float)
for i in range (T2s_min, T2s_max):
    T2p = np.append(T2p, 1/((1000/i)-1/T2))
    
#Quadratic phase shift's Period
period = [3,4,5,6,7,8,9,10,11,12]

for Period in period: 
    phi = []
    for i in range (0,Nr):
        p = np.deg2rad(QuadraticPhase(Period,i))
        phi.append(p)

    ##-------------------------------------Simulation Section -------------------------------------------
    #---------------------------------Reference Noise------------------------------
    #For different Period RF spoiling, Noise added to the F-state is: 1/sqr(N)*5%*F0 with T2* = 33ms, TR = 6ms
    TR_ref = 6e-3
    TR_list_ref = [TR_ref*1e3] * Nr
    TE_ref = TR_ref/2
    T2s_ref = 33
    T2p_ref = 1/((1000/T2s_ref)-1/T2)

    off_resonance_f, M_transverse =  blochc(off_res, off_res_samplesize, M0 = M0 , alpha = alpha, phi = phi, dphi = dphi, TR= TR_list_ref, TE= TE_ref*1e3, T1 = T1*1e3, T2 = T2*1e3)
    F_magnitude_inhomo, F_state = AddfieldInhomogeneous(M_transverse, TR_ref, TE_ref, T2p_ref, off_resonance_f)

    plt.plot(F_state, np.absolute(F_magnitude_inhomo[0,:]))
    plt.xlim(-25, 25)
    F0_ref = np.abs(F_magnitude_inhomo[0,np.where(F_state == int(2))])

    Noise_sd = 1/np.sqrt(Period)*F0_ref*0.05
    Noise = np.random.normal(0,Noise_sd,(testnumber,off_res_samplesize)) + 1j*np.random.normal(0,Noise_sd,(testnumber,off_res_samplesize))


    #---------------------------------Simulation---------------------------------
    ErrorMeanfigdata = np.asarray(np.zeros((1,np.shape(T2p)[0])), dtype = float)
    Errorstdfigdata = np.asarray(np.zeros((1,np.shape(T2p)[0])), dtype = float)
    Biasfigdata = np.asarray(np.zeros((1,np.shape(T2p)[0])), dtype = float)
    Meanfigdata = np.asarray(np.zeros((1,np.shape(T2p)[0])), dtype = float)
    #All_Data = np.asarray([], dtype = float) 
    number = 0

    for tr in TR:
        #Simulate Data
        #Generate bSSFP off-resonance profile using Bloch simulation 
        TR_list = [tr *1e3] * Nr 
        TE = tr/2

        off_resonance_f, M_transverse =  blochc(off_res, off_res_samplesize, M0 = M0 , alpha = alpha, phi = phi, dphi = dphi, TR= TR_list, TE= TE*1e3, T1 = T1*1e3, T2 = T2*1e3)
        off_resonance_f = off_resonance_f/2
        # plt.figure(2)
        # plt.plot(off_resonance_f, np.abs(M_transverse))

        #Add Field inhomogeneous and Noise
        #Data is an Array contain (T2*num, noise num, off-res f num)
        Data, T2Star_GT = quadratic_RFspoiling_bSSFP (T1, T2, int(1/tr), tr, tr/2, tip_angle, Period, M0, Nr, M_transverse, off_resonance_f, T2p, testnumber, AddNoise, Noise)


        #np.save("TwelvePeriodData_"+str(tr)+"_0.01Noise.npy", Data)
        #Data = np.load("./DataNpy/SixPeriod_0.01/SixPeriodData_"+str(round(tr, 2-int(floor(log10(abs(tr))))-1))+"_.npy")

        #---------------------------------Fit T2*---------------------------------
        Mean, Errorstd, Bias = T2StarFit(Data, T2Star_GT, Period, tr)
        #figurex = DrawT2StarGraph(Data, T2Star_GT, Period, tr)
        #Store data for the map
        Mean = Mean.reshape(1,np.shape(T2Star_GT)[0])
        ErrorMean = Mean-T2Star_GT
        Errorstd = Errorstd.reshape(1,np.shape(T2Star_GT)[0])
        Bias = Bias.reshape(1,np.shape(T2Star_GT)[0])
        ErrorMeanfigdata = np.append(ErrorMeanfigdata, ErrorMean, axis = 0)
        Errorstdfigdata = np.append(Errorstdfigdata, Errorstd, axis = 0)
        Biasfigdata = np.append(Biasfigdata, Bias, axis = 0)
        Meanfigdata = np.append(Meanfigdata, Mean, axis = 0)


        print(tr)
    ErrorMeanfigdata = np.delete(ErrorMeanfigdata,0,0)
    Errorstdfigdata = np.delete(Errorstdfigdata,0,0)
    Biasfigdata = np.delete(Biasfigdata,0,0)
    Meanfigdata = np.delete(Meanfigdata,0,0)        

    #Save Result
    np.save("./Figure/"+str(Period)+"Period_0.05Noise_"+str(TR_n)+"TRs/"+str(Period)+"PeriodData_0.05Noise_ErrorMean.npy", ErrorMeanfigdata)
    np.save("./Figure/"+str(Period)+"Period_0.05Noise_"+str(TR_n)+"TRs/"+str(Period)+"PeriodData_0.05Noise_Errorstd.npy", Errorstdfigdata)
    np.save("./Figure/"+str(Period)+"Period_0.05Noise_"+str(TR_n)+"TRs/"+str(Period)+"PeriodData_0.05Noise_Bias.npy", Biasfigdata)
    np.save("./Figure/"+str(Period)+"Period_0.05Noise_"+str(TR_n)+"TRs/"+str(Period)+"PeriodData_0.05Noise_Mean.npy", Meanfigdata)