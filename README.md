# ECG_DSS_CAE

A Decision Support System (DSS) tool to speed up the clinical annotation process in 15-lead ECG signals. It uses a CAE to find all anomalous windows in a given ECG signal.
CAE trained on syntheticand real ECG signals from the PTB Diagnostic ECG database 1.0.

- pretrained model using only synthetic data: DpNet.dill/DPNet_cpu.dill
- pretrained model using synthetic and real data: model.pt
- train data available upon request
- test data will be available soon 


Dataset used:
PTB Diagnostic ECG database 1.0: https://www.physionet.org/content/ptbdb/1.0.0/

Installation requires the anaconda software.
For the installation use given setup files. 

HOW TO USE:

conda activate ECGDSS
python ecg_gui_main.py


WORK IN PROGRESS...