#!/bin/bash 
@echo off

echo "Preparing a new conda enviroment called ECGDSS"
conda activate.bat 
conda create -n ECGDSS python=3.7
conda install -n ECGDSS pip
python create_config_file.py 

echo "Installing libraries"
conda activate ECGDSS

pip install plotly 
pip install neurokit2
pip install dash
pip install dill
pip install ipywidgets>=7.0.0
pip install wfdb
call pip install pytorch-lightning

if [[$(lshw -C display | grep vendor) =~ Nvidia ]]; then 
	conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
else
	conda install pytorch torchvision torchaudio cpuonly -c pytorch 
fi