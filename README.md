# Final_Year_Project
This project aim at understanding and analysis the N-periodic ka-SPGR T2* mapping performance. (parameters specifically adjust for Parkinson's Disease detection) 

Code/N_Periodic_kaSPGR/N_periodic_ka_SPGR_MonteCarloSimulation
1. Monte Carlo Simulation is performed for ka-SPGR with different period and TRs, within Parkinson's Disease Substantia Nigra T2* value changed range (13ms-53ms). 
    - The Python file and Jupyter notebook version of the simulation are both provided. 
    - A intuitive display for the result is proved in "MonteCarloResultDisplay.ipynb".
    - A step-py-step illustration of the simulation is also provided as "N_periodic_kaSPGR_ModelStepByStepIllustration.ipynb".
    - Monte Carlo Simulation result (mean, percentage bias, standard deviation) and Maps are stored as 2D array in SimulationResult.
2. Optimised scan parameters are suggested by the simulation result, for TR > 6ms (scanner limit):
    - 7-periodic ka-SPGR with TR = 6ms
    - 12-periodic ka-SPGR with TR = 6ms

Code/N_Periodic_kaSPGR/PhantomExperiment
1. Phantom experiment is performed on a  NIST/ISMRM Premium System Phantom Model (SN:130-102) using the optimised ka-SPGR sequence and compare with GOLD-standard Multi-echo GRE. 
    - All acquired data are provided in folder "Data_dicom"
2. code for reconstruct the T2* mapping are provided here. 
    - Mask is generated to acquire only voxels in fiducial spheres using "GenerateMask.ipynb"
    - T2* calculation of the acquired data are performed in "GoldStandard_Multi_echo_FLASH_T2StarMapping.ipynb" and "NPeriodic_kaSPGR_T2StarMapping.ipynb"
    - Analysis, T2* mapping images are generated in "ReportFigures.ipynb" and "ResultDisplay.ipynb", and stored in RESULT
