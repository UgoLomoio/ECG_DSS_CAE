# ECG_DSS_CAE

A Decision Support System (DSS) tool to speed up the clinical annotation process in 12-lead ECG signals. It uses a CAE to find all anomalous windows in a given ECG signal.
CAE trained on synthetic and real ECG signals from the PTB Diagnostic ECG database 1.0 and PTB-XL dataset.

Model available:
- pretrained model using only synthetic ECGs: DpNet.dill/DPNet_cpu.dill
- pretrained model finetuned using real ECG data from PTB: model.pt -> TESTED ON PTB 
- pretrained model finetuned using real ECG data from PTB-XL: model.pt -> TESTED ON CPSC2018

Data availability:

- PTB Diagnostic ECG database 1.0: https://www.physionet.org/content/ptbdb/1.0.0/
- PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
- China Physiological Signal Challenge 2018: http://2018.icbeb.org/Challenge.html
- train and test split used is available upon request


HOW TO INSTALL

1. Installation requires the anaconda software. 

2. Create a python 3.8 conda enviroment and install pip inside it

3. Use: pip install -r requirements


HOW TO USE

1. After the installation, open the command prompt and type: 

		conda activate "enviroment_name"

2. Then, type: 
	
		python ecg_gui_main.py

Cite this work:


