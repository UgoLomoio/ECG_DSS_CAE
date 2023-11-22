# ECG_DSS_CAE

A Decision Support System (DSS) tool to speed up the clinical annotation process in 15-lead ECG signals. It uses a CAE to find all anomalous windows in a given ECG signal.
CAE trained on syntheticand real ECG signals from the PTB Diagnostic ECG database 1.0.

- pretrained model using only synthetic data: DpNet.dill/DPNet_cpu.dill
- pretrained model using synthetic and real data: model.pt
- train data available upon request
- test data will be available soon 


Dataset used:
PTB Diagnostic ECG database 1.0: https://www.physionet.org/content/ptbdb/1.0.0/

HOW TO INSTALL

1. Installation requires the anaconda software. 

2. After the installation of anaconda python, open the command prompt and use one of the given setup files (depending on your operative system) to automatically install python libraries. 

	On Windows:

		setupWindows.bat

	One Linux or Mac: 
		
		setupLinuxMac.sh
		
	If you are using a Linux or Mac system, you may need to change the permissions of the setupLinuxMac.sh file. To do so, open the command prompt and type: 
		chmod +x setupLinuxMac.sh

	If the installation is successful, you should see a new conda environment called ECG_DSS in your anaconda navigator.

	If an error occours during the installation, try to open the setup file with a text editor and install the libraries manually using the command prompt.

HOW TO USE

1. After the installation, open the command prompt and type: 

		conda activate ECG_DSS

2. Then, type: 
	
		python ecg_gui_main.py

Cite this work:


