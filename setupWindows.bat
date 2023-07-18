@echo off

SET onlycpu=true

FOR /F "tokens=* USEBACKQ" %%F IN (`wmic path Win32_VideoController get Name`) DO (
	
	echo %%F
	SET var=%%F
	echo %var%
	
	ECHO.%var% | FIND /V "NVIDIA">Nul && (SET onlycpu=false)

)

echo "Preparing a new conda enviroment called ECGDSS"
call activate.bat 
call conda create -n ECGDSS python=3.8
call conda install -n ECGDSS pip

echo "Installing libraries"
call conda activate ECGDSS 
call pip install plotly 
call pip install neurokit2
call pip install dash
call pip install dill
call pip install ipywidgets>=7.0.0
call pip install wfdb
call pip install pandas
call pip install pytorch-lightning

if onlycpu==false(call conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge) else (call conda install pytorch torchvision torchaudio cpuonly -c pytorch)