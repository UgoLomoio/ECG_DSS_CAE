"""
*************************************************************************************************************
*                                                                                                           *
*           User Interface developed by Ugo Lomoio at Magna Graecia University of Catanzaro                 *
*            DPNET Model Developed by Salvatore Petrolo @t0re199 at University of Calabria                  *
*                                                                                                           *
*************************************************************************************************************
"""

"""
Utility libraries
"""
from EcgStuffs.src.dpnet import dpnet_loader                        #load the model
from EcgStuffs.src.windows.WindowingUtils import sliding_window     #extract windows from a signal
#from EcgStuffs.src.pickleio.PickleIO import load_object             #load .data files 

import csv                                                          #load .csv files 
import os                                                                                   
import numpy as np

from tkinter.filedialog import askopenfile

import torch                                                        #the model is written with pythorc and use torch functions
import torch.nn.functional as F                                     #compute MSE error between orginal signal and reconstructed signal

import matplotlib.pyplot as plt 

def load_data():
        """
        Load ECG data. 
        Format Supported:
        * .data
        * .csv

        ECG data must be:
        * sampling frequency 500 Hz
        * number of channels 15            
        """

       
        file_path = askopenfile(mode='r', filetypes=[('ECG Data Files', ['*data', '*csv'])]) 
        if file_path is None:
            return None 
        else:
        
            if file_path is not None:
                extension = os.path.splitext(file_path.name)[1]
                basename = os.path.basename(file_path.name)
            
                if extension == ".data":
                    filename = basename[:len(basename)-5]
                    signal = load_object(basename, os.path.dirname(file_path.name))
                    nchs_data, n_record = signal.shape
                elif extension == ".csv": 
                    filename = basename[:len(basename)-4]
                    with open(file_path.name, 'r') as csvdata:
                        sniffer = csv.Sniffer()
                        dialect = sniffer.sniff(csvdata.readline())
                        delimiter = dialect.delimiter
                    signal = np.genfromtxt(file_path.name, delimiter = delimiter, dtype = np.float32)
                    nchs_data, n_record = signal.shape
                    
                if nchs_data != 15:
                    if n_record == 15:
                        print("Transposing the given signal.")
                        signal = signal.T
                    else:
                        signal = None
                        print("This model is trained using 15 channel ECG 500Hz data. Given a {} channel ECG data.".format(nchs_data))
            
                return signal
            else:
                print("Please Upload a valid .data file before click the Start Button.")
                return None 

 
def printProgressBar(iteration, total):

    length = 100
    fill = '█'
    printEnd = '\r'
    percent = ("{0:." + str(2) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f"\r |{bar}| Reconstruction. Current progress: {percent}%", end = printEnd)

    if percent == 100:
        print()

def reconstruction_run(model, signal):
        
    """
    reconstruction with progress bar.
    """

    nchs, n_record = signal.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = sliding_window(signal, int(0x5 * 0.8 * 0x1F4), stride=int(0.2 * 0x1F4))
    windows = torch.from_numpy(np.array(windows))
        
    interval = 1500
    n_win = windows.shape[0]
        
    #progress bar
    idx = 0
    windows_batch_data = windows.clone().detach().unsqueeze(dim=0x1).to(device)
    reconstruction_errors = []
    reconstructions = []
    original = []
    printProgressBar(0, n_win)
    for i in range(0, n_win, int(interval/100)):

        start = idx*interval 
        end = interval + idx*interval
        idx += 1
        batch_data = windows_batch_data[i, 0, :, :].reshape((1, 1, 15, 2000)).to(device)
        with torch.no_grad():

            reconstruction = model(batch_data)
            reconstructions.append(reconstruction[0, 0, :, :interval])
            original.append(batch_data[0, 0, :, :interval])
            reconstruction_errors.append((F.mse_loss(reconstruction[0, 0, :, :interval], batch_data[0, 0, :, :interval])).item())
            #recon_np = (reconstruction.cpu().detach().numpy()).reshape(1, nchs, 2000)
            if end > n_record:
                printProgressBar(n_win, n_win)
                
            else:
                printProgressBar(i, n_win)
        torch.cuda.empty_cache()

    printProgressBar(n_win, n_win)       
    return original, reconstructions, reconstruction_errors

def plot_anomaly(original, reconstruction, reconstruction_error, anomaly_id, time):

    fig = plt.figure(figsize = (15, 15))
    plt.title("Anomaly Number {} with reconstruction error {}".format(anomaly_id, reconstruction_error))
    plt.xlabel("Seconds")
    plt.ylabel("mV")
    plt.plot(time, original[0, :], 'g')
    plt.plot(time, reconstruction[0, :], 'r--')
    plt.legend()
    plt.show()

def main():

    print("ECG + PPG Anomaly Detector tool.\n")
   
    if not torch.cuda.is_available(): #DPNET model is trained using torch cuda with NVIDIA GPU. 
                                      #DPNET model can work now even with a machine without NVIDIA GPU and torch cuda installed. 

        print("Cannot import torch cuda. Make sure your computer have an NVIDIA GPU and you have installed torch cuda.")
        print("Converting the model to a CPU model.\n")
        model = dpnet_loader.load_cpu()
    else:

        print("Cuda is available\n")
        model = dpnet_loader.load()
        #create_cpu_model(DPNET_PATH, DPNET_CPU_PATH)

    while True:
        print("Please choose one ECG+PPG 15 channels and 500 Hz file to analyze. \n")
        while True:
            signal = load_data()
            if signal is not None:
                break 
    
       
        original, reconstructions, reconstruction_errors = reconstruction_run(model, signal)

        min_ = min(reconstruction_errors)
        max_ = max(reconstruction_errors)
        while True:
            print("\nThreshold value for anomaly detection must be much greater then {} but minor then {}\n".format(min_, max_))
            threshold = float(input("Enter a threshold value for anomaly detection: "))
            if threshold < max_ and threshold > min_:
           
                anomalies = []
                time = []
                for i, recon_error in enumerate(reconstruction_errors):
                    if recon_error > threshold:
                        anomalies.append(reconstructions[i])      
                        start_time = (i)*1500/500
                        end_time = (i+1)*1500/500
                        step = (end_time-start_time)/1500
                        time.append(np.arange(start_time, end_time, step))
                print("\n Found {} anomalies. \n".format(len(anomalies)))
                break

        while True: 
            print("\n Select the anomaly to visualize. \n")
            anomaly_id = input("Select the anomaly to visualize between {} and {}. Or Enter X to end the analysis. ".format(0, len(anomalies)-1))
            if anomaly_id.lower() == "x":
                break
            else:
                anomaly_id = int(anomaly_id)
                if anomaly_id < len(anomalies):
                
                    plot_anomaly(original[anomaly_id].cpu().detach().numpy(), reconstructions[anomaly_id].cpu().detach().numpy(), reconstruction_errors[anomaly_id], anomaly_id, time[anomaly_id])
                
        choice = input("\n Press 0 if you don't want to analyze another signal. ")
        if choice == '0':
            break

if __name__ == '__main__':
    main()