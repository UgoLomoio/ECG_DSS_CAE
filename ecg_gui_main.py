"""
*************************************************************************************************************
*                                                                                                           *
*   DPNet Model testing, version on real data + Decision Support System developed by Ugo Lomoio             *
*                       at Magna Graecia University of Catanzaro                                            *
*                                                                                                           *
*   DPNET Model on syntethic data Developed by Salvatore Petrolo @t0re199 at University of Calabria         *
*                                                                                                           *
*************************************************************************************************************
"""

import tkinter as tk
from ecg_gui_class import ECG_GUI 
import torch #Leave here

#UNCOMMENT TO CREATE THE CPU MODEL 
#from EcgStuffs.src.dpnet.dpnet_loader import create_cpu_model
#from EcgStuffs.src.dpnet.conf import DPNET_PATH, DPNET_CPU_PATH

#HOW TO INSTALL TORCH CUDA WITH ANACONDA: conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

def main():

    window = tk.Tk()
    window.geometry("1100x1000+0+0")

    if not torch.cuda.is_available(): #DPNET model is trained using torch cuda with NVIDIA GPU. 
                                      #DPNET model can work now even with a machine without NVIDIA GPU and torch cuda installed. 

        print("Cannot import torch cuda. Make sure your computer have an NVIDIA GPU and you have installed torch cuda.")
        print("Converting the model to a CPU model.")

    else:

        print("Cuda is available")
        #create_cpu_model(DPNET_PATH, DPNET_CPU_PATH)

    ECG_GUI(window)
    window.mainloop()

if __name__ == '__main__':
    main()
