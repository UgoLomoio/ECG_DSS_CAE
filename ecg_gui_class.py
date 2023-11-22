"""
*************************************************************************************************************
*                                                                                                           *
*   Model test and Decision Support System developed by Ugo Lomoio at Magna Graecia University of Catanzaro *
*   DPNET Model Developed by Salvatore Petrolo @t0re199 at University of Calabria                           *
*                                                                                                           *
*************************************************************************************************************
"""

from statistics import mode
import numpy as np

"""
First GPU libraries: for upload a ECG file, read the file, extract single beats and find anomalies in the signal
"""
import tkinter as tk 
from tkinter import ttk 
from tkinter.filedialog import askopenfile
import wfdb

from threading import Thread

"""
Second GPU libraries: Visualize and annotate the ECG signal and the anomalies
"""
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
#from cefpython3 import cefpython as cef

"""
Utility libraries
"""
from EcgStuffs.src.dpnet import dpnet_loader                        #load the model
from EcgStuffs.src.windows.WindowingUtils import sliding_window     #extract windows from a signal
#from EcgStuffs.src.pickleio.PickleIO import load_object             #load .data files 
import neurokit2 as nk                                              #extract r peaks from signal
import csv                                                          #load .csv files 
import os                                                                                   
import platform
import json                                                         #json dumps needed in dash

import torch                                                        #the model is written with pythorc and use torch functions
import torch.nn.functional as F                                     #compute MSE error between orginal signal and reconstructed signal
import webbrowser
       

#from dash.dependencies import ClientsideFunction

torch.cuda.empty_cache()

# Fix for PyCharm hints warnings
#WindowUtils = cef.WindowUtils()

# Platforms
WINDOWS = (platform.system() == "Windows")
LINUX = (platform.system() == "Linux")
MAC = (platform.system() == "Darwin")

app_dash = Dash(__name__)          #create the dash app fist 

"""
SOME GLOBAL VARIABLES: need the use of global variables because we need to share them between the two different GUIs (two different classes: Dash and ECG_GUI)
"""

#number of clicks current of buttons
max_manual_n_clicks = 0                     #manual update
max_annotate_n_clicks = 0                   #annotate point
max_prec_n_clicks = 0                       #prec anomaly
max_next_n_clicks = 0                       #next anomaly
max_save_n_clicks = 0                       #save html
max_anomalies_n_clicks = 0                  #compute anomalies
max_export_n_clicks = 0                     #export annotations
max_reload_n_clicks = 0                     #reload
max_beat_prec_n_clicks = 0                  #prec beat
max_beat_next_n_clicks = 0                  #next beat


text_error_label = ""
anomalies_idx = []                         #array of anomaly ids founded
anomaly_id = None                          #current anomaly id visualized
curr_threshold = None                      #threshold to compute anomalies
reconstruction_all = None                  #reconstructed signal
reconstruction_errors = []                 #array of MSE reconstruction errors
annotations = None                         #array of text annotations
annotated_points = []                      #array of points text annotated
fig_global = None                          #figure currently visualized
signal = None                              #ECG signal
nchs = 15                                  #number of channels in the signal
curr_chan = 0                              #current channel visualized, from 1 to 15, 0 for "all channels"
curr_xmin = 0                              #current xmin slider value
curr_xmax = None                           #current xmax slider value
curr_beat = None                           #current beat visualized 
colors = ["green", "white", "blue", "lightcoral", "orange", "cyan", "magenta", "yellow", "purple", "hotpink", "midnightblue", "lime", "olive", "gold", "palegreen"] #channel colors
filename = "ECG_file"                      #current filename

max_channel_n_clicks = [0]*(nchs+1)        #array of number of clicks of single channels and all channels buttons
change_figure_button_clicked = False       #True if manual update button is clicked
change_figure_anomalies_computed = False   #True if compute anomalies button is clicked
change_figure_annotation = False           #True if annotate points button is clicked
change_figure_prec_beat = False            #True if prec beat button is clicked
change_figure_next_beat = False            #True if next beat button is clicked
change_reload = False                      #True if reload button is clicked

fig_clicked = None                         #temporary figure that display the user clicked button in red
prec_clicked = None                        #precedent clicked point by the user
prec_show_recon_error = "No"               #precedent "show recon error mode" radio button value: yes or no
show_recon_error = "No"                    #current "show recon error mode" radio button value
fig_all = None                             #global figure to display when "All Channels" button is clicked
fig_single_ch = [None]*nchs                #array of global figures to display when "Fig X" button is clicked
time = None                                #array of time in seconds
fs = 500                                   #model is trained with ECG data 500Hz frequency sampling
rpeaks_all = None                          #array of rpeaks founded
prec_mode = "Select"                       #precedent "interaction mode": Select or Draw
config = {"modeBarButtonsToAdd": []}
isreset = False                            #true when user click the reset button
end_chs = False 
end_all = False
first_cef = True
channels_names = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6", "aVR", "aVL", "aVF", "X", "Y", "Z"]



"""
Preprocessing functions for the ECG signal
"""
from scipy.signal import butter, lfilter, iirnotch, resample

def bandpass(lowcut, highcut, order=3, fs = 500):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(cutoff, q, fs=500):
    
    nyq = 0.5*fs
    freq = cutoff/nyq
    b, a = iirnotch(freq, q)
    return b, a

def myfilter(lowcut, highcut, powerline, data):
    
    nchs = 12
    filtered_data = np.zeros_like(data)
    for ch in range(nchs):
        ch_data = data[:, ch]
        b, a = bandpass(lowcut, highcut)   
        x = lfilter(b, a, ch_data)
        f, e = notch_filter(powerline, 30)
        z = lfilter(f, e, x) 
        filtered_data[:, ch] = (z)
    return filtered_data

def multichannel_resample(signal, new_lenght):
    nchs = 15
    resampled_signal = np.zeros((new_lenght, nchs))
    for ch in range(nchs):
        resampled_signal[:, ch] = resample(signal[:, ch], new_lenght)
        #print(resampled_signal[:, ch].shape, resample(signal[:, ch], new_fs).shape)
    return resampled_signal



"""
In a Dash App we can only implement one callback function. 
We need now some global function to call in a unique big callback. 
"""

def plot_one_ch(ch, row=1):

    """
    Function that create the plot of the single channel "ch" given.
    """
    global signal 
    global filename 
    global colors 
    global reconstruction
    global fig_single_ch 
    global fig_all
    global show_recon_error
    global time 
    global fs 
    global rpeaks_all 
    global channels_names 

    i = ch - 1
    if fig_single_ch[i] is None:
        
        print("create fig channel ", ch)
        nchs, n_record = signal.shape
        if time is None:
            x = np.arange(0, n_record, dtype=float)
            time = x / fs
        
        fig_single_ch[i] = go.FigureWidget(layout = dict(clickmode='event', hovermode = 'closest', newshape=dict(fillcolor="rgba(255, 0, 0, 0.2)", line=dict(color="rgba(255,255,255, 1.0)", width=5, dash = 'dashdot')),
                                           updatemenus = [
                                                            dict(
                                                                type = "buttons", 
                                                                buttons = list([
                                                                dict(
                                                                    label = 'Select Point',
                                                                    method = 'relayout',
                                                                    args = [{'clickmode': 'event', 'hovermode': 'closest'}]
                                                                    )
                                                                ]), 
                                                                x=0.5,
                                                                xanchor="center",
                                                                yanchor="bottom", 
                                                                showactive=False
                                                            )
                                                        ]
                                           ))
        channel = signal[i, :]
        ch_name = channels_names[i]
        if show_recon_error == "Yes":    #TO SLOW FOR NOW 
            fig_single_ch[i].add_trace(go.Scatter(
                x=time, 
                y=channel,
                fill=None,
                mode='markers+lines',
                line_color = colors[i],
                line_width = 3, 
                marker_color = colors[i],
                marker_size = 3, 
                name=ch_name
            ))
        
            fig_single_ch[i].add_trace(go.Scatter(
                x=time,
                y=reconstruction_all[i, :],
                fill='tonexty',
                mode='none',
                fillcolor='pink',
                name="reconstruction error ch {}".format(ch_name)
            ))

        else: #elif show_recon_error == "No":
            fig_single_ch[i].add_trace(go.Scattergl(
                x=time, 
                y=channel,
                fill=None,
                mode='markers+lines',
                line_color = colors[i],
                line_width = 3, 
                marker_color = colors[i],
                marker_size = 3, 
                name = ch_name

            ))

        fig_single_ch[i].add_trace(go.Scattergl(
                x=time, 
                y=reconstruction_all[i, :],
                fill=None,
                mode='lines',
                line_color = 'red', 
                line_width = 3, 
                line_dash = 'dot',
                name="reconstruction {}".format(ch_name)
        ))
        if rpeaks_all is not None:
            arrow_list_ch = []       
            distances = []
            y_rr = []
            x_rr = []
            for j, rpeak in enumerate(rpeaks_all):
                if j + 1 < len(rpeaks_all):
                    next_rpeak = rpeaks_all[j+1]
                    print(rpeak, next_rpeak)
                    dist = r_r_distance(rpeak, next_rpeak)
                    distances.append(dist)
                    middle = rpeak + int((next_rpeak - rpeak)/2)
                    y_middle = max(fig_single_ch[i].data[0]["y"][rpeak], fig_single_ch[i].data[0]["y"][next_rpeak])
                    y_rr.append(y_middle)
                    x_rr.append(middle/fs)
            
                scatter_ch = fig_single_ch[i].data[0]
                t = scatter_ch['x'][rpeak] 
                y = scatter_ch['y'][rpeak]
                arrow=dict(x=t, y=y, xref="x", yref="y", text="R", showarrow=False, font={"color": "red"},  yshift = 30)
                arrow_list_ch.append(arrow)


            fig_single_ch[i].add_trace(go.Scattergl(
                x=x_rr, 
                y=y_rr,
                fill=None,
                mode='lines+text',
                line_color = 'red', 
                line_width = 1, 
                line_dash = 'dot',
                text = distances,
                name="r-r peak distance",
                textposition = "top center"
            ))

            fig_single_ch[i].update_layout(annotations=arrow_list_ch)

        fig_single_ch[i].update_layout(        
                                xaxis = {
                                      'autorange': True,
                                      'constrain': 'range',
                                      'range': [0, time[-1]],
		      			              'rangeslider': {'visible': False, 'autorange': True, 'range': [0, time[-1]]},
                                      'type': 'linear',
                                      'showspikes': True,
                                      'showticklabels': True,
                                      'spikemode': 'across+marker',
                                      'title_text': 'Time [s]',
                                      'uirevision': True
                                },
                                yaxis = {
                                      'autorange': True,
                                      'title_text':'mV',
                                      'title_standoff':0.1
                                },
                                #gridcolor='LightPink'
                                template="plotly_dark",
                                title_text = "{}: Channel {}".format(filename, ch_name)
                        )

    return fig_single_ch[i]

def plot_all_chs():

    """
    Function that create the plot of all channels.
    Calls plot_single_ch for all 15 channels.
    """
    global nchs 
    global signal 
    global colors 
    global fig_all 
    global fig_single_ch
    global fig_global
    global fs 
    global time 
    global channels_names

    if fig_all is None:

        
        print("create fig all channels")

        fig_all = go.FigureWidget(layout = dict( clickmode='event', hovermode = 'closest', newshape=dict(fillcolor="rgba(255, 0, 0, 0.2)", line=dict(color="rgba(255,255,255, 1.0)", width=1, dash = 'dashdot')),
                                  updatemenus = [
                                              dict(
                                                    type = "buttons", 
                                                    buttons = list([
                                                        dict(
                                                            label = 'Select Point',
                                                            method = 'relayout',
                                                            args = [{'clickmode': 'event', 'hovermode': 'closest'}]
                                                            )
                                                        ]),
                                                    x=0.5, 
                                                    xanchor="center", 
                                                    yanchor="bottom", 
                                                    showactive=False
                                              )]
                                  )).set_subplots(rows = nchs, cols=1, shared_xaxes=True)
        

        n_record = signal.shape[1]
        if time is None:
            x = np.arange(0, n_record, dtype=float)
            time = x / fs
        channel_color_dict = {}
        for i in range(nchs):
            
            if fig_single_ch[i] is None:
                ch = i+1
                fig_single_ch[i] = plot_one_ch(ch)

            if i == 0:

                x_axis_name = "xaxis"
            else:

                x_axis_name = "xaxis{}".format(i+1)

            channel = signal[i, :]
            ch_name = channels_names[i]
            channel_color_dict[ch_name] = [colors[i]]*n_record
            fig_all.add_trace(go.Scattergl(x=time, y=channel, mode='markers+lines', visible = True, line= {'color':colors[i], 'width':3}, marker = {'color': colors[i], 'size': 3}, name=ch_name), row=i+1, col=1)
            layout_args = {}
            layout_args["{}.showticklabels".format(x_axis_name)] = False
            layout_args['{}.showspikes'.format(x_axis_name)] = True
            layout_args['{}.spikemode'.format(x_axis_name)] = 'across+marker'
            fig_all.update_layout(layout_args)   

        fig_all.update_layout(      xaxis = {
                                              'autorange': True,
                                              'constrain': 'range',
                                              'range': [0, time[-1]],       
		        			                  'rangeslider': {'visible': False, 'autorange': True, 'range': [0, time[-1]]},
                                              'type': 'linear',
                                              'showspikes': True,
                                              'spikemode': 'across+marker',
                                              'uirevision': True
                                    },
                                    xaxis15 = {
                                        'title_text': 'Time [s]',
                                        "showticklabels": True
                                    },
                                    yaxis8 = {
                                              'title_text': 'mV',
                                              'title_standoff':0.1
                                    },
                                    template="plotly_dark",
                                    title_text = "{}: 15 Channels ECG".format(filename)
                       )
    return fig_all

def annotate_rpeaks(window=None, pb=None, label=None):
   
    """
    Function that add text annotations to figures: fig_global, fig_all and fig_single_ch to annotate rpeaks found by neurokit2.
    """
    global rpeaks_all
    global fig_all
    global fig_global 
    global fig_single_ch
    global nchs 
    global isreset 
    
    arrow_list_chs = []
    for j in range(nchs):

        arrow_list_ch = []       
        distances = []
        y_rr = []
        x_rr = []
        for i, rpeak in enumerate(rpeaks_all):
            if i + 1 < len(rpeaks_all):
                next_rpeak = rpeaks_all[i+1]
                dist = r_r_distance(rpeak, next_rpeak)
                distances.append(dist)
                middle = rpeak + int((next_rpeak - rpeak)/2)
                y_middle = max(fig_single_ch[j].data[0]["y"][rpeak], fig_single_ch[j].data[0]["y"][next_rpeak])
                y_rr.append(y_middle)
                x_rr.append(middle/fs)
            
            scatter_ch = fig_single_ch[j].data[0]
            t = scatter_ch['x'][rpeak] 
            y = scatter_ch['y'][rpeak]
            arrow=dict(x=t, y=y, xref="x", yref="y", text="R", showarrow=False, font={"color": "red"},  yshift = 30)
            arrow_list_ch.append(arrow)

        fig_single_ch[j].add_trace(go.Scattergl(
            x=x_rr, 
            y=y_rr,
            fill=None,
            mode='lines+text',
            line_color = 'red', 
            line_width = 1, 
            line_dash = 'dot',
            text = distances,
            name="r-r peak distance",
            textposition = "top center"
        ))
        fig_single_ch[j].add_trace(go.Scattergl(
            x=x_rr, 
            y=y_rr,
            fill=None,
            mode='lines+text',
            line_color = 'red', 
            line_width = 1, 
            line_dash = 'dot',
            text = distances,
            name="r-r peak distance",
            textposition = "top center"
        ))
        arrow_list_chs.append(arrow_list_ch)

    arrow_list_all = [] 
    for i, rpeak in enumerate(rpeaks_all):
        print("annotate rpeak ", rpeak)

        for j in range(nchs):

            if j == 0:
                xref = "x"
                yref = "y"
            else:
                xref = "x{}".format(j+1)
                yref = "y{}".format(j+1)
            
            scatter_all = fig_all.data[j]
            t = scatter_all['x'][rpeak]
            y = scatter_all['y'][rpeak]
            
            arrow=dict(x=t, y=y, xref=xref, yref=yref, text="R", showarrow=False, font={"color": "red", "size": 12}, yshift = 3)
            if j == 0:    #ANNOTATE ONLY THE FIRST CHANNEL TO MAKE THE TOOL FASTER
                arrow_list_all.append(arrow)

    fig_all.update_layout(annotations=arrow_list_all)  

    if window is not None and pb is not None and label is not None:
        for j in range(nchs):
            if not isreset:
                fig_single_ch[j].update_layout(annotations=arrow_list_chs[j])
                   
                pb["value"] = round(((j + 1) / nchs) * 100, 2)
                label['text'] = "Finding Rpeaks... Current progress: {}%".format(pb["value"])
                pb.grid(row = 3, column = 2, padx=(80, 10))
                label.grid(row = 4, column = 2, padx=(100, 30))
                window.update()
                arrow_list_chs.append(arrow_list_ch)
            else:
                break
    if curr_chan == 0:
       fig_global = go.FigureWidget(fig_all)
    else:
       fig_global = go.FigureWidget(fig_single_ch[curr_chan-1])

def create_fig(selectedpoint = None):

    """
    Function that create a figure when a point is clicked by the user, selected point is highlighted in red.
    If no point is selected, return fig_all or fig_single_ch (if they are None, recreate them).
    """
    #plot ECG
    global signal 
    global colors 
    global filename 
    global fig_all
    global fig_single_ch
    global curr_chan
    global fig_global 
    global annotated_points
    global annotations
    global fs 

    nchs, n_record = signal.shape
    
    if selectedpoint is not None:

        print("create_fig, chan ", curr_chan, " selected point ", selectedpoint)
        if curr_chan == 0:

            fig = go.FigureWidget(fig_global)#fig_all
            
            for i in range(nchs):
            
                scatter = fig.data[i]
                x_sel =  [scatter['x'][selectedpoint]]
                y_sel =  [scatter['y'][selectedpoint]]
                if i == 0:
                    fig.add_trace(go.Scattergl(x=x_sel, y=y_sel, mode='markers', marker = {'color': 'red', 'size': 6}, name="Clicked Point"), row=i+1, col=1)
                else:
                    fig.add_trace(go.Scattergl(x=x_sel, y=y_sel, mode='markers', marker = {'color': 'red', 'size': 6}, showlegend=False), row=i+1, col=1)

        elif curr_chan <=15:
            
            fig = go.FigureWidget(fig_global)#fig_single_ch[curr_chan-1]
            scatter = fig.data[0]
            x_sel =  [scatter['x'][selectedpoint]]
            y_sel =  [scatter['y'][selectedpoint]]
            fig.add_trace(go.Scattergl(x=x_sel, y=y_sel, mode='markers', marker = {'color': 'red', 'size': 6}, name="Clicked Point"))
        
        half_window = 750/fs
        if (x_sel[0]-half_window)<0:
            xmin = 0
        else:
            xmin = x_sel[0]-half_window
    
        if (x_sel[0]+half_window)>n_record:
            xmax = n_record
        else:
            xmax = x_sel[0] + half_window
        x_range = [xmin, xmax]
        fig = set_x_range(fig, x_range)
        #fig_global = fig 
        return fig 

    else:

        if curr_chan == 0:
            fig_all = plot_all_chs()
            fig_global = fig_all
            """
                if annotations is not None and len(annotated_points) > 0:
                    for annotated_point in range(len(annotated_points)):
                        annotation = annotations[annotated_point]
                        print(annotated_point, annotation)
                        for i in range(nchs):
        
                            if i == 0:
                                xref = "x"
                                yref = "y"
                            else:
                                xref = "x{}".format(i+1)
                                yref = "y{}".format(i+1)

                            scatter = fig_global.data[i]
                            x =  scatter['x'][annotated_point]
                            y =  scatter['y'][annotated_point]
                            fig_global.add_annotation(x=x, y=y, text = annotation, xref=xref, yref=yref, showarrow=True, arrowhead=7, font={"color": "red"}, arrowcolor="red")
            """
        elif curr_chan <= 15:
            for i in range(nchs):
                fig_single_ch[i] = plot_one_ch(i+1)
                if i == curr_chan - 1:
                    fig_global = fig_single_ch[i]
            """ 
                if annotations is not None and len(annotated_points) > 0:
                    for annotated_point in range(len(annotated_points)):
                        annotation = annotations[annotated_point]
                        scatter = fig_global.data[0]
                        x =  scatter['x'][annotated_point]
                        y =  scatter['y'][annotated_point]
                        fig_global.add_annotation(x=x, y=y, text = annotation, xref='x', yref='y', showarrow=True, arrowhead=7, font={"color": "red"}, arrowcolor="red")
            """ 
    return fig_global
                

def sigle_channel_button_handler(ch, button_n_clicks,  max_channelX_n_clicks):
    """
    Function that handles "Ch *ch*" buttons, where *ch* can be a number between 1 and 15 (example: if *ch* == 1, the button "Ch 1" is clicked and this function handles the creation and display
                                                                                                   of a new updated figure that show only the selected channel 1). 
    """
    global max_channel_n_clicks
    global curr_chan
    global fig_global 

    if button_n_clicks > max_channelX_n_clicks: #new click

        max_channel_n_clicks[ch] = button_n_clicks  
        curr_chan = ch
        fig_global = create_fig()
        return fig_global
    
    else:
        
        return None

def channels_button_handler(channel_button_names, channel_button_n_clicks):
    """
    Function that handles all "Ch *ch*" buttons, where *ch* can be a number between 1 and 15. 
    """
    global nchs
    global max_channel_n_clicks 
    global curr_chan 
    global prec_chan 
    global fig_global 

    prec_chan = curr_chan 
    for i, button_name in enumerate(channel_button_names):
        channel_button_n_click = channel_button_n_clicks[i]
        max_channelX_n_clicks = max_channel_n_clicks[i]
        fig = sigle_channel_button_handler(i, channel_button_n_click, max_channelX_n_clicks) #return None if the button is not clicked, else return new figure
        if fig is not None: #new click
            break

    return fig #can be None

def find_rpeaks(signal, window = None):

    """
    Function that find rpeaks in a ECG signal using neurokit2 library.
    """
    global fs
    global nchs 

    rpeaks_all = nk.ecg_findpeaks(signal[0, :].T, sampling_rate=fs)
    rpeaks_all = rpeaks_all['ECG_R_Peaks']
            
    #for ch in range(nchs):
    #    cleaned_signal, waves_all = nk.ecg_delineate(signal[ch, :].T, rpeaks_all, sampling_rate=fs, method="dwt", show=False, show_type='all')
    #    print(waves_all)

    if window is not None:
        rpeaks_window = []
        for rpeak in rpeaks_all:
            if rpeak in window:
                rpeaks_window.append(rpeak)
        #same for waves
        return rpeaks_window
    else:
        return rpeaks_all

def r_r_distance(rpeak1, rpeak2):

    """
    Computing R-R peaks distance between two consecutive beats.
    """

    if rpeak1 > rpeak2:
        distance = (rpeak1 - rpeak2)/fs
    else:
        distance = (rpeak2 - rpeak1)/fs

    return round(distance, 2) 

def compute_frequency(curr_beat):

    """
    Compute frequency (beats per minutes, bpm) of the current window visualized by the user. 
    """
    #we compute beat per minutes using a window of 11 consecutive beats: 1 central beat, 5 precedents,5 successive
    global time
    global rpeaks_all 

    n_beats = 11

    if curr_beat - 5 > 0:

        xmin = time[rpeaks_all[curr_beat-5]]
    
    else:

        xmin = 0

    if curr_beat + 5 <len(rpeaks_all):

        xmax = time[rpeaks_all[curr_beat+5]]
    
    else:

        xmax = time[-1]
       
    time_m = (xmax-xmin)*(10/600) #from seconds to minutes
    freq = n_beats/time_m

    return round(freq, 2)

def manual_update_plot(xmin, xmax, anomaly_id_input):

    """
    Function that update the plot using the parameters: xmin, xmax and anomaly_id given by the user usin a text box. 
    """
    global text_error_label
    global anomaly_id
    global anomalies_idx
    global fs 
    global fig_global 

    if xmin is not None:
        xmin = int(xmin)
    if xmax is not None:
        xmax = int(xmax)
    if anomaly_id_input != "" and anomaly_id is not None:
       anomaly_id_input = int(anomaly_id_input)

    error = False 
    if anomaly_id_input != "" and anomaly_id is not None:
           
        if anomaly_id_input < 0 or anomaly_id_input > len(anomalies_idx)-1:
            text_error_label += "anomaly_id must be > 0 and < {} but given {}. ".format(len(anomalies_idx), anomaly_id_input)
            error = True
        if not error:
            anomaly_id = anomaly_id_input
            anomaly = anomalies_idx[anomaly_id]/fs
            half_window = 750/fs
            if (anomaly-half_window)<0:
                xmin = 0
            else:
                xmin = anomaly - half_window - half_window
                
            n_record = signal.shape[1]
            if (anomaly+half_window)>n_record:
                xmax = n_record
            else:
                xmax = anomaly + half_window + half_window

            x_range = [xmin, xmax]
            fig_global = set_x_range(fig_global, x_range)
            
    elif ((xmin is not None) and (xmax is not None)):
        #check parameter consistency
        if xmin > xmax:
            text_error_label += "xmin must be <= of xmax. Given xmin = {} and xmax = {}. ".format(xmin, xmax)
            error = True
        if xmax > time[-1]:
            xmax = time[-1]
        if xmin < 0:
            xmin = 0
        if not error:
            x_range = [xmin, xmax]
            fig_global = set_x_range(fig_global, x_range)
    
    if error:
        text_error_label_to_return = "{}".format(text_error_label) #copy
        text_error_label=""
        return text_error_label_to_return, False, fig_global
    else:
        return "", True, fig_global

def set_x_range(fig, x_range):
    """
    Function that set the current range of the signal to visualize.
    """
    #set xmin e xman slider manually
    global curr_xmin
    global curr_xmax

    curr_xmin = x_range[0] 
    curr_xmax = x_range[1]
    fig.update_layout(xaxis = {'autorange': False, 'range': x_range, 'uirevision': True, 'rangeslider': {'autorange': True, 'range': x_range}})
    return fig 

def go_prec_anomaly():

    """
    Function used to visualize the precedent anomaly.
    """
    global anomaly_id
    global anomalies_idx
    global signal 
    global fig_global
    global reconstruction_errors
    global fs 
    global curr_xmax
    global curr_xmin 

    if anomaly_id-1 >= 0:

        print(anomaly_id, " -> ", anomaly_id-1)
        anomaly_id -= 1
        prec_anomaly = anomalies_idx[anomaly_id]/fs
        half_window = 750/fs
        if (prec_anomaly-half_window)<0:
            xmin = 0
        else:
            xmin = prec_anomaly - half_window - half_window
        n_record = signal.shape[1]
        if (prec_anomaly+half_window)>n_record:
            xmax = n_record
        else:
            xmax = prec_anomaly + half_window + half_window

        x_range = [xmin, xmax] 
        curr_xmax = xmax
        curr_xmin = xmin 
        fig_global = set_x_range(fig_global, x_range)
    
    return fig_global

def go_next_anomaly():

    """
    Function used to visualize the next anomaly.
    """

    global anomaly_id
    global anomalies_idx
    global signal 
    global fig_global
    global reconstruction_errors
    global fs 
    global curr_xmax 
    global curr_xmin 

    if anomaly_id+1 < len(anomalies_idx):
        
        print(anomaly_id, " -> ", anomaly_id+1)
        anomaly_id += 1
        next_anomaly = anomalies_idx[anomaly_id]/fs
        half_window = 750/fs
        if (next_anomaly-half_window)<0:
            xmin = 0
        else:
            xmin = next_anomaly - half_window  - half_window
        
        n_record = signal.shape[1]
        if (next_anomaly+half_window)>n_record:
            xmax = n_record
        else:
            xmax = next_anomaly + half_window + half_window
        
        x_range = [xmin, xmax] 
        curr_xmax = xmax
        curr_xmin = xmin 
        fig_global = set_x_range(fig_global, x_range)

    return fig_global

def annotate_fig_single_chs(annotation, ind):

    """
    Function used to apply the text annotation given by the user to the 15 figures in the array "fig_single_ch".
    """
    global nchs 
    global fig_single_ch
    global fig_global
    global curr_chan
    global fs 
    global signal 
    global end_chs

    for j in range(nchs):     
            
        print("ch ", j+1)
        scatter_ch = fig_single_ch[j].data[0]
        x = scatter_ch['x'][ind] 
        y = scatter_ch['y'][ind]
        fig_single_ch[j].add_annotation(x=x, y=y, xref="x", yref="y", text=annotation, showarrow=True, arrowhead=1, font={"color": "red"}, arrowcolor="red")
        if curr_chan == j+1:
            fig_global.add_annotation(x=x, y=y, xref="x", yref="y", text=annotation, showarrow=True, arrowhead=1, font={"color": "red"}, arrowcolor="red")
        
            half_window = 750/fs
            if (x - half_window)<0:
                xmin = 0
            else:
                xmin = x - half_window
       
            n_record = signal.shape[1]
            if (x + half_window)>n_record:
                xmax = n_record
            else:
                xmax = x + half_window
            x_range = [xmin, xmax] 
            fig_global = set_x_range(fig_global, x_range)

    end_chs = True


def reannotate_fig(fig):

    """
    Function used to reannotate the fig .
    """


    if annotations is not None and len(annotated_points) > 0:
        for annotated_point in range(len(annotated_points)):
            annotation = annotations[annotated_point]
            scatter = fig.data[0]
            x =  scatter['x'][annotated_point]
            y =  scatter['y'][annotated_point]
            fig.add_annotation(x=x, y=y, text = annotation, xref='x', yref='y', showarrow=True, arrowhead=7, font={"color": "red"}, arrowcolor="red")
    return fig

def annotate_fig_all(annotation, ind):

    """
    Function used to apply the text annotation given by the user to the figure with all 15 channels "fig_all".
    """

    global nchs 
    global fig_global
    global fig_all 
    global curr_chan
    global fs 
    global signal 
    global end_all

    for j in range(nchs): 

        print("all ", j+1)
        
        if j == 0:
            xref = "x"
            yref = "y"
        else:
            xref = "x{}".format(j+1)
            yref = "y{}".format(j+1)
            
        scatter_all = fig_all.data[j]
        x = scatter_all['x'][ind]
        y = scatter_all['y'][ind]

        #if j == 0:
        fig_all.add_annotation(x=x, y=y, xref=xref, yref=yref, text=annotation, showarrow=True, arrowhead=1, arrowcolor="red", font={"color": "red", "size": 12})
        
        if curr_chan == 0:
            fig_global.add_annotation(x=x, y=y, xref=xref, yref=yref, text=annotation, showarrow=True,  arrowhead=1, arrowcolor = "red", font={"color": "red", "size": 12})

    half_window = 750/fs
    if (x - half_window)<0:
        xmin = 0
    else:
        xmin = x - half_window
       
    n_record = signal.shape[1]
    if (x + half_window)>n_record:
        xmax = n_record
    else:
        xmax = x + half_window
    x_range = [xmin, xmax] 
    fig_global = set_x_range(fig_global, x_range)
    end_all = True

    print("end all")

def annotate(annotation, ind, recon_error = False, add = True):

    """
    Function used to apply text annotation given by the user to all the figures. Uses a parallel approach with two threads: one calls annotate_fig_all(annotation, ind), 
                                                                                                                            the other calls annotate_fig_single_ch(annotation, ind)
    """
    global nchs 
    global annotations 
    global annotated_points
    global fig_all
    global curr_chan
    global fig_global 
    global fig_single_ch
    global fs 
    global end_chs
    global end_all

    print("annotate ", annotation, " clicked ", ind, " curr_chan ", curr_chan)

    if not recon_error:
        
        if add:
            annotations[ind] = annotation
            annotated_points.append(ind)
        """
        for i in range(nchs):
            
            print(2, i)
            if i == 0:
                xref = "x"
                yref = "y"
            else:
                xref = "x{}".format(i+1)
                yref = "y{}".format(i+1)
            
            scatter = fig_all.data[i]
            x =  scatter['x'][ind]
            y =  scatter['y'][ind]
            if curr_chan == 0:
                if fig_global is None:
                    fig_global = go.FigureWidget(fig_all)
                fig_global.add_annotation(x=x, y=y, text = annotation, xref=xref, yref=yref, showarrow=False, font={"color": "red", "size": 9}, arrowcolor="red")
            fig_all.add_annotation(x=x, y=y, text = annotation, xref=xref, yref=yref, showarrow=False, font={"color": "red", "size": 9}, arrowcolor="red")
            
            scatter_ch = fig_single_ch[i].data[0]
            x = scatter_ch['x'][ind]
            y = scatter_ch['y'][ind]
            if curr_chan>=1 and curr_chan <= 15 and i == curr_chan - 1:
                if fig_global is None:
                    fig_global = go.FigureWidget(fig_single_ch[curr_chan-1])
                fig_global.add_annotation(x=x, y=y, text = annotation, xref='x', yref='y', showarrow=True, arrowhead=7, font={"color": "red"}, arrowcolor="red")
            fig_single_ch[i].add_annotation(x=x, y=y, text = annotation, xref='x', yref='y', showarrow=True, arrowhead=7, font={"color": "red"}, arrowcolor="red")
        """#SLOW

        #create to threads to speed up the annotation process
        end_all = False
        end_chs = False
        thread_chs = Thread(target=annotate_fig_single_chs, args = [annotation, ind])
        thread_chs.start()
        thread_all = Thread(target=annotate_fig_all, args=[annotation, ind])
        thread_all.start()

        while True:
            if end_all and end_chs: #check if two threads ended
                break

    else: 

        scatter_ch = fig_global.data[0]
        x = scatter_ch['x'][ind]
        half_window = 750
        ys = scatter_ch['y'][ind-half_window:ind+half_window]
        y = max(ys)+0.2
        if y > 2:
            y = 2
        fig_global.add_annotation(x=x, y=y, text = annotation, xref='x', yref='y', showarrow=False, font={"color": "red", "size": 12})
       
    return fig_global, json.dumps([ind, annotation])

def color_clicked(ind):
    
    """
    Function that create a figure when a point is clicked by the user, selected point is highlighted in red.
    """
    global curr_chan 
    print("clicked ", ind, " chan ", curr_chan)
    fig_updated = create_fig(ind)
    print("end fig creation")
    return fig_updated

def export_annotations():

    """
    Function that allow the user to export text annotations done in a .csv file". Maybe add import_annotations button and functionality in future.
    """
    global annotated_points
    global annotations
    global filename
    
    path = os.getcwd()+os.sep+"annotations"+os.sep+filename+"_annotations.csv"
    i = 0
    while True:
        if os.path.isfile(path):
            path = os.getcwd()+os.sep+"annotations"+os.sep+filename+"_annotations_{}.csv".format(i)
            i += 1
        else: 
            break

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["annotated_point", "user_annotation"]
        writer.writerow(header)
        for annotated_point in annotated_points:
            annotation = annotations[annotated_point]
            row = [annotated_point, annotation]
            writer.writerow(row)
    

def save_html():

    """
    Function that allow the user to save the current session in a .html file.
    Check if the user can choose the directory.
    """
    global fig_global
    global filename 

    path = os.getcwd() + os.sep + "HTML_plots" + os.sep + filename + ".html"
    fig_global.write_html(path)
  
def empty_figure():
    
    """
    Creates an empty figure.
    Used for two clicks update figure strategy:
        One click to update the figure,
        Second click to visualize the updated figure.
    """
    fig = go.FigureWidget()
    fig.update_layout(          xaxis = {
                                              'autorange': True,
                                              'constrain': 'range',       #maybe we can set directly here curr_xmin and curr_xmax, without set_x_range method 
		        			                  'rangeslider': {'visible': False, 'autorange': True},
                                              'type': 'linear',
                                              'showspikes': True,
                                              'spikemode': 'across+marker',
                                },
                                yaxis = {
                                              'autorange': True
                                },
                                template="plotly_dark",
                                title_text = "Double click the button to update the figure..."
                     )
    return fig 

'''
app_dash.clientside_callback(
    """
    function(myPlot, x, fig_all.data[j]['y'][fig_global.clickData] for j in range nchs]) {
        
        var select_button = {
            name: 'Select Point',
            icon: Plotly.Icons.pencil,
            direction: 'up',
            click: function (gd) {

                Plotly.relayout(gd, { newshape: { line: { color: "red" }, fillcolor: "red", opacity: 0.3 }, clickmode: 'select', hovermode: 'closest', dragmode: 'select', paper_bgcolor: "black", plot_bgcolor: "black"});

            }
        };
        
        text_annotations = myPlot.layout.annotations || [];
        
        var annotation = window.prompt("Clicked point: " + x + ". \nEnter the text annotation to add: ");
        
        for y in range ys:
            add_annotation = {
                    text: annotation,
                    x: x,
                    y: y,
                    showarrow: true,
                    font: {
                        family: 'Courier New, monospace',
                        size: 16,
                        color: 'red'
                    },
                    align: 'center',
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 2,
                    arrowcolor: 'red',
                    bordercolor: '#c7c7c7',
                    borderwidth: 2,
                    borderpad: 4,
                    bgcolor: '#ff7f0e',
                    opacity: 0.8
                    };

                text_annotations.push(add_annotation);
                Plotly.relayout(myPlot, { annotations: text_annotations });
           }
        });
    }
    """,
    Output('fig', 'figure'),
    Input('fig', 'figure'),
    Input('fig', 'clickData')
)
'''

@app_dash.callback(

        Output("error", "children"),
        Output("error", "hidden"),
        Output("fig", "figure"),
        Output("clicked-annotation-label", "children"), 
        Output("anomaly-label", "children"),
        Output("beat-label", "children"),
        Output("fig", "config"),

        Input("annotation", "value"),
        Input("fig", "clickData"),
        Input("xmin", "value"),
        Input("xmax", "value"),
        Input("anomaly", "value"),
        Input("annotate-button", "n_clicks"),
        Input("manual-update-button", "n_clicks"),
        Input("next-button", "n_clicks"),
        Input("prec-button", "n_clicks"),
        Input("beatnext-button", "n_clicks"),
        Input("beatprec-button", "n_clicks"),
        Input("export-button", "n_clicks"),
        Input("reload-button", "n_clicks"),
        Input("show-recon-radio", "value"),
        Input("allchannels-button", "n_clicks"), 
        Input("channel1-button", "n_clicks"),
        Input("channel2-button", "n_clicks"),
        Input("channel3-button", "n_clicks"),
        Input("channel4-button", "n_clicks"),
        Input("channel5-button", "n_clicks"),
        Input("channel6-button", "n_clicks"),
        Input("channel7-button", "n_clicks"),
        Input("channel8-button", "n_clicks"),
        Input("channel9-button", "n_clicks"),
        Input("channel10-button", "n_clicks"),
        Input("channel11-button", "n_clicks"),
        Input("channel12-button", "n_clicks"),
        Input("channel13-button", "n_clicks"),
        Input("channel14-button", "n_clicks"),
        Input("channel15-button", "n_clicks"),
        Input("save-button", "n_clicks"),
        Input("anomalies-button","n_clicks"),
        Input("rangeslider", "value"),
        Input("threshold", "value"),
        prevent_initial_call=True
)
def update_plot_callback(annotation, clickData, xmin, xmax, anomaly_id_input, annotate_n_clicks, manual_n_clicks, next_n_clicks, 
                         prec_n_clicks, next_beat_n_clicks, prec_beat_n_clicks, export_n_clicks, reload_n_clicks, show_reconerror, 
                         all_ch_n_clicks, ch1_n_clicks, ch2_n_clicks, ch3_n_clicks, ch4_n_clicks, ch5_n_clicks, 
                         ch6_n_clicks, ch7_n_clicks, ch8_n_clicks, ch9_n_clicks, ch10_n_clicks, ch11_n_clicks, ch12_n_clicks,
                         ch13_n_clicks, ch14_n_clicks, ch15_n_clicks, save_n_clicks, anomalies_n_clicks, x_range, threshold):

    """
    Main callback of the app.
    Handles all the buttons and functionality to make the visualization of the ECG signal interactive. 
    List of supported functionalities:
        1. Text annotation (Annotate Point)
        2. Single channel visualization (Ch 1, ..., Ch 15) 
        3. All channels visualization (All channels)
        4. Compute anomalies and navigate next anomaly or prec anomaly.
        5. Clicked point (when Mode: Select and the user clicks a specific point in the signal)
        6. Shape annotation (when Mode: Draw and the user clicks the Reload button)
        7. Show reconstructed signal in Single channel visualization 
        8. Show reconstruction errors and : Yes or No
        9. Show current beat visualized, navitate next beat or prec beat and show current frequency bpm.
        10. Show rpeaks and R-R distances.
        11. Save current session in HTML format.
        12. Save current annotations in .csv format.
        13. Zoom, Pan, Interactive visualization.
        14. Save a screenshot.

    """
    
    global max_manual_n_clicks
    global max_next_n_clicks
    global max_prec_n_clicks
    global max_annotate_n_clicks
    global max_save_n_clicks
    global max_anomalies_n_clicks
    global max_export_n_clicks
    global max_reload_n_clicks
    global max_beat_prec_n_clicks
    global max_beat_next_n_clicks

    global anomaly_id 
    global anomalies_idx
    global reconstruction_errors
    global fig_global
    global signal 
    global max_channel_button_n_clicks
    global curr_xmax
    global curr_xmin 
    global curr_chan 
    global change_figure_button_clicked
    global curr_threshold
    global change_figure_anomalies_computed
    global change_figure_annotation
    global change_figure_prec_beat
    global change_figure_next_beat
    global change_reload
    global fig_clicked 
    global prec_clicked 
    global prec_show_recon_error
    global show_recon_error
    global fig_all
    global fig_single_ch
    global annotations
    global annotated_points
    global curr_beat 
    global prec_mode 
    global config 

    print("callback_global ", curr_chan)
    ind = None 
    if clickData is not None:
        ind = int(clickData["points"][0]["pointNumber"])

    #channel buttons handler
    channel_button_n_clicks = [all_ch_n_clicks, ch1_n_clicks, ch2_n_clicks, ch3_n_clicks, ch4_n_clicks, ch5_n_clicks, 
                               ch6_n_clicks, ch7_n_clicks, ch8_n_clicks, ch9_n_clicks, ch10_n_clicks, ch11_n_clicks, 
                               ch12_n_clicks, ch13_n_clicks, ch14_n_clicks, ch15_n_clicks]
    channel_button_names = ["allchannels-button", "channel1-button", "channel2-button", "channel3-button", "channel4-button",
                            "channel5-button", "channel6-button", "channel7-button", "channel8-button", "channel9-button",
                            "channel10-button", "channel11-button", "channel12-button", "channel13-button", "channel14-button",
                            "channel15-button"]

    curr_beat_label = json.dumps("Visualize next/prec beat")
        
    if len(anomalies_idx) == 0 or anomaly_id is None:
        
        text_anomaly_label = json.dumps("Threshold must be an float number between {} and {}".format(round(min(reconstruction_errors), 5), round(max(reconstruction_errors), 5)))
    else:

        text_anomaly_label = json.dumps("Found {} anomalies with threshold {}. Current Anomaly: {}/{}".format(len(anomalies_idx), curr_threshold, anomaly_id+1, len(anomalies_idx)))
    

    config = {'modeBarButtonsToAdd': ['drawline', 'drawrect', 'eraseshape'], 'responsive': True, 'displayModeBar': True}

    #dash x slider handler 
    dash_x_min = x_range[0]
    dash_x_max = x_range[1]
    changed_x_range = False
    if curr_xmin != dash_x_min:
        curr_xmin = dash_x_min
        changed_x_range = True
    if curr_xmax != dash_x_max:
        curr_xmax = dash_x_max
        changed_x_range = True

    #compute anomalies 
    if show_recon_error != show_reconerror:
      
        prec_show_recon_error = show_recon_error
        show_recon_error = show_reconerror
        if show_recon_error != prec_show_recon_error:
            print("show reconstruction error: ", show_recon_error)
            for i in range(nchs):
                ch = i+1
                fig_single_ch[i] = None
                fig_single_ch[i] = plot_one_ch(ch)
            if curr_chan > 0 and curr_chan <= 15:
                fig_global = fig_single_ch[curr_chan-1]
                #readd annotations 
                if len(annotated_points) > 0:
                    for i, annotated_point in enumerate(annotated_points):
                        annotation = annotations[annotated_point]
                        fig_global, _ = annotate(annotation, annotated_point, add=False)
                if rpeaks_all is not None:
                    annotate_rpeaks()
                if len(anomalies_idx)>0:
                    for i, recon_error in enumerate(reconstruction_errors):
                        if recon_error >= curr_threshold:
                            full_window = 1500
                            half_window = full_window/2
                            x_anomaly = int(half_window + i*full_window)
                            anomalies_idx.append(x_anomaly)
                            fig_global = annotate("Reconstruction Error: {}".format(round(recon_error, 5)), x_anomaly, True)[0] 
                if ind is not None:
                    fig_clicked = color_clicked(ind)
                    fig_clicked = set_x_range(fig_clicked, [curr_xmin, curr_xmax])
                    for i, rpeak in enumerate(rpeaks_all):
                        if i+1 < len(rpeaks_all):
                            next_rpeak = rpeaks_all[i+1]
                            if ind >= rpeak and ind < next_rpeak:
                                curr_beat = i
                                break
                            elif ind > max(rpeaks_all):
                                curr_beat = len(rpeaks_all)-1
                                break 
                            elif ind < min(rpeaks_all):
                                curr_beat = 1

                    freq = compute_frequency(curr_beat)
                    curr_beat_label = json.dumps("Current Beat: {}/{}. Frequency: {} bpm".format(curr_beat+1, len(rpeaks_all), freq))
                    return json.dumps(""), True, fig_clicked, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
                else:
                    fig_global = set_x_range(fig_global, [curr_xmin, curr_xmax])
                    return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
       
    #second click button
    #manual update
    elif change_figure_button_clicked: 
        
        print("change figure cause button clicked ", curr_chan)
        fig_button_handler = channels_button_handler(channel_button_names, channel_button_n_clicks) #can be None if no channel button is pressed 
        change_figure_button_clicked = False
        if curr_beat is not None:
            freq = compute_frequency(curr_beat)
            curr_beat_label = json.dumps("Current Beat: {}/{}. Frequency: {} bpm".format(curr_beat+1, len(rpeaks_all), freq))

        if fig_clicked is not None:
            return json.dumps(""), True, fig_clicked, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
        else:
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
    
    #compute anomalies
    elif change_figure_anomalies_computed:
        if anomalies_n_clicks > max_anomalies_n_clicks:
            
            max_anomalies_n_clicks = anomalies_n_clicks
            change_figure_anomalies_computed = False
            freq = compute_frequency(curr_beat)
            curr_beat_label = json.dumps("Current Beat: {}/{}. Frequency: {} bpm".format(curr_beat+1, len(rpeaks_all), freq))
            print("change figure cause button 'Compute Anomalies' is clicked")
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 

    #Annotate point
    elif change_figure_annotation:
        if annotate_n_clicks > max_annotate_n_clicks:

            print("change figure cause button 'Annotate Point' is clicked")
            max_annotate_n_clicks = annotate_n_clicks
            change_figure_annotation = False
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
    
    #Prec beat
    elif change_figure_prec_beat:
        if prec_beat_n_clicks > max_beat_prec_n_clicks:

            print("change figure cause button 'Prec Beat' is clicked")
            freq = compute_frequency(curr_beat)
            curr_beat_label = json.dumps("Current Beat: {}/{}. Frequency: {} bpm".format(curr_beat+1, len(rpeaks_all), freq))
            max_beat_prec_n_clicks = prec_beat_n_clicks
            change_figure_prec_beat = False
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
  
    #Next beat
    elif change_figure_next_beat:
        if next_beat_n_clicks > max_beat_next_n_clicks:

            print("change figure cause button 'Next Beat' is clicked")
            freq = compute_frequency(curr_beat)
            curr_beat_label = json.dumps("Current Beat: {}/{}. Frequency: {} bpm".format(curr_beat+1, len(rpeaks_all), freq))
            max_beat_next_n_clicks = next_beat_n_clicks
            change_figure_next_beat = False
            fig_clicked = color_clicked(ind)
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
    
    #When the user use the dash slider to set x_range
    elif changed_x_range:
        print("here")
        if ind is None:
            print("changed x_range to [{}, {}]".format(curr_xmin, curr_xmax))
            #fig_global = reannotate_fig(fig_global)
            fig_global = set_x_range(fig_global, x_range)
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config
        else:
            if ind in annotated_points:
                print("changed x_range to [{}, {}] 2".format(curr_xmin, curr_xmax))
                fig_global = reannotate_fig(fig_global)
                fig_global = set_x_range(fig_global, x_range)
                return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config
            else:
                print("else")
                #fig_clicked = reannotate_fig(fig_clicked)
                fig_clicked = set_x_range(fig_clicked, x_range)
                return json.dumps(""), True, fig_clicked, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 

    #Reload
    elif change_reload:
        if reload_n_clicks > max_reload_n_clicks:
            print("change figure cause 'Reload' is clicked")
            max_reload_n_clicks = reload_n_clicks
            change_reload = False
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 

    else:

        #first click
        fig_button_handler = channels_button_handler(channel_button_names, channel_button_n_clicks) #can be None if no channel button is pressed 

        #reset all double click buttons 
        change_figure_prec_beat = False
        change_figure_next_beat = False
        change_figure_annotation = False 
        change_figure_anomalies_computed = False 
        change_figure_button_clicked = False
        change_reload = False
       

        #Reload 
        if reload_n_clicks > max_reload_n_clicks:
            
            print("reload")
            max_reload_n_clicks = reload_n_clicks
            change_reload = True
            return json.dumps(""), True, empty_figure(), json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config

        #compute anomalies with user given threshold 
        elif anomalies_n_clicks > max_anomalies_n_clicks:

            print("compute anomalies")
            if curr_chan == 0: #force single channel view
                curr_chan = 1
                fig_global = go.FigureWidget(fig_single_ch[0])
            else:
                fig_global = go.FigureWidget(fig_single_ch[curr_chan-1])
               
            max_anomalies_n_clicks = anomalies_n_clicks

            if threshold is not None and threshold != "":
                #if float(threshold) != curr_threshold:
                    threshold = float(threshold)
                    anomalies_idx = []
                    curr_threshold = threshold
                    print("threshold: ", curr_threshold)
                    for i, recon_error in enumerate(reconstruction_errors):
       
                        if recon_error >= curr_threshold:
                           full_window = 1500
                           half_window = full_window/2
                           x_anomaly = int(half_window + i*full_window)
                           anomalies_idx.append(x_anomaly)
                           fig_global = annotate("Reconstruction Error: {}".format(round(recon_error, 5)), x_anomaly, True)[0]  
                        
                    anomaly_id = 0
                    text_error_label_to_return, hidden, fig_global = manual_update_plot(None, None, anomaly_id)         
                    change_figure_anomalies_computed = True
                    text_anomaly_label = json.dumps("Found {} anomalies with threshold {}. Current Anomaly: {}/{}".format(len(anomalies_idx), curr_threshold, anomaly_id+1, len(anomalies_idx)))
                

            if len(anomalies_idx) > 0:
                text_anomaly_label = json.dumps("Found {} anomalies with threshold {}. Current Anomaly: {}/{}".format(len(anomalies_idx), curr_threshold, anomaly_id+1, len(anomalies_idx)))
            else:
                text_anomaly_label = json.dumps("Found 0 anomalies with threshold {}.".format(curr_threshold))
            
            for i, rpeak in enumerate(rpeaks_all):
                window_center = curr_xmin*fs + ((curr_xmax*fs- curr_xmin*fs)/2)
                if i+1 < len(rpeaks_all):
                    next_rpeak = rpeaks_all[i+1]
                    if window_center >= rpeak and window_center < next_rpeak:
                        curr_beat = i
                        break
                    elif window_center > max(rpeaks_all):
                        curr_beat = len(rpeaks_all)-1
                        break 
                    elif window_center < min(rpeaks_all):
                        curr_beat = 0
                        break 

            freq = compute_frequency(curr_beat)
            curr_beat_label = json.dumps("Current Beat: {}/{}. Frequency: {} bpm".format(curr_beat+1, len(rpeaks_all), freq))

            return json.dumps(""), True, empty_figure(), json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
        
        #new point clicked
        elif prec_clicked != ind :
        
            print("clicked point: ", ind)
            fig_clicked = color_clicked(ind)   
            fig_clicked = set_x_range(fig_clicked, [curr_xmin, curr_xmax])

            for i, rpeak in enumerate(rpeaks_all):
                if i+1 < len(rpeaks_all):
                    next_rpeak = rpeaks_all[i+1]
                    if ind >= rpeak and ind < next_rpeak:
                        curr_beat = i
                        break
                    elif ind > max(rpeaks_all):
                        curr_beat = len(rpeaks_all)-1
                        break 
                    elif ind < min(rpeaks_all):
                        curr_beat = 0
                        break
            freq = compute_frequency(curr_beat)
            curr_beat_label = json.dumps("Current Beat: {}/{}. Frequency: {} bpm".format(curr_beat+1, len(rpeaks_all), freq))
            
            prec_clicked = ind 
            print("end clicked")
            return json.dumps(""), True, fig_clicked, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
        
        #Annotation point
        elif annotate_n_clicks > max_annotate_n_clicks:

            print("annotate ", annotation, " clicked point ", ind)
            max_annotate_n_clicks = annotate_n_clicks
            change_figure_annotation = True
            fig_global, annotated_and_clicked_label = annotate(annotation, ind) 
            return json.dumps(""), True, empty_figure(), annotated_and_clicked_label, text_anomaly_label, curr_beat_label, config 
        
        #Manual Update
        elif manual_n_clicks > max_manual_n_clicks:

            print("manual update: ", xmin, xmax, anomaly_id_input)
            max_manual_n_clicks = manual_n_clicks  

            if len(anomalies_idx) == 0 or anomaly_id_input is not None or anomaly_id_input != "":
                
                text_error_label_to_return, not_hidden, fig_global = manual_update_plot(xmin, xmax, None)
                text_anomaly_label = json.dumps("Compute the anomalies first, threshold must be an float number between {} and {}".format(round(min(reconstruction_errors), 5), round(max(reconstruction_errors), 5)))
            else:
                
                text_error_label_to_return, not_hidden, fig_global = manual_update_plot(None, None, anomaly_id_input)
                text_anomaly_label = json.dumps("Found {} anomalies with threshold {}. Current Anomaly: {}/{}".format(len(anomalies_idx), curr_threshold, anomaly_id+1, len(anomalies_idx)))

            return json.dumps(text_error_label_to_return), not_hidden, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
        
        #Next anomaly
        elif next_n_clicks > max_next_n_clicks:

            print("next anomaly")
            max_next_n_clicks = next_n_clicks
            if len(anomalies_idx) == 0 or anomaly_id is None:
                text_anomaly_label = json.dumps("Compute the anomalies first, threshold must be an float number between {} and {}".format(round(min(reconstruction_errors), 5), round(max(reconstruction_errors), 5)))
            else:
                if anomaly_id+1 < len(anomalies_idx):
                    fig_global = go_next_anomaly()
                    recon_error = reconstruction_errors[anomaly_id]
                    full_window = 1500
                    half_window = full_window/2
                    x_anomaly = int(half_window + anomaly_id*full_window)
                                
                    fig_global = annotate("Reconstruction Error: {}".format(round(recon_error, 5)), x_anomaly, True)[0] 
                    text_anomaly_label = json.dumps("Found {} anomalies with threshold {}. Current Anomaly: {}/{}".format(len(anomalies_idx), curr_threshold, anomaly_id+1, len(anomalies_idx)))
                else:
                    text_anomaly_label = json.dumps("Out of range anomaly ids. Current Anomaly: {}/{}".format(anomaly_id+1, len(anomalies_idx)))
            
            for i, rpeak in enumerate(rpeaks_all):
                window_center = curr_xmin*fs + ((curr_xmax*fs- curr_xmin*fs)/2)
                if i+1 < len(rpeaks_all):
                    next_rpeak = rpeaks_all[i+1]
                    if window_center >= rpeak and window_center < next_rpeak:
                        curr_beat = i
                        break
                    elif window_center > max(rpeaks_all):
                        curr_beat = len(rpeaks_all)-1
                        break 
                    elif window_center < min(rpeaks_all):
                        curr_beat = 0
                        break 

            freq = compute_frequency(curr_beat)
            curr_beat_label = json.dumps("Current Beat: {}/{}. Frequency: {} bpm".format(curr_beat+1, len(rpeaks_all), freq))
            
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 

        #Prec anomaly
        elif prec_n_clicks > max_prec_n_clicks:

            print("prec anomaly")
            max_prec_n_clicks = prec_n_clicks
            if len(anomalies_idx) == 0 or anomaly_id is None:
                text_anomaly_label = json.dumps("Compute the anomalies first, threshold must be an float number between {} and {}".format(round(min(reconstruction_errors), 5), round(max(reconstruction_errors), 5)))
            else:
                if anomaly_id-1 >= 0:
                    fig_global = go_prec_anomaly()
                    recon_error = reconstruction_errors[anomaly_id]
                    full_window = 1500
                    half_window = full_window/2
                    x_anomaly = int(half_window + anomaly_id*full_window)
                    fig_global = annotate("Reconstruction Error: {}".format(round(recon_error, 5)), x_anomaly, True)[0] 
                    text_anomaly_label = json.dumps("Found {} anomalies with threshold {}. Current Anomaly: {}/{}".format(len(anomalies_idx), curr_threshold, anomaly_id+1, len(anomalies_idx)))
                else:
                    text_anomaly_label = json.dumps("Out of range anomaly ids. Current Anomaly: {}/{}".format(anomaly_id+1, len(anomalies_idx)))
            
            for i, rpeak in enumerate(rpeaks_all):
                window_center = curr_xmin*fs + ((curr_xmax*fs - curr_xmin*fs)/2)
                if i+1 < len(rpeaks_all):
                    next_rpeak = rpeaks_all[i+1]
                    if window_center >= rpeak and window_center < next_rpeak:
                        curr_beat = i
                        break
                    elif window_center > max(rpeaks_all):
                        curr_beat = len(rpeaks_all)-1
                        break 
                    elif window_center < min(rpeaks_all):
                        curr_beat = 0
                        break 

            freq = compute_frequency(curr_beat)
            curr_beat_label = json.dumps("Current Beat: {}/{}. Frequency: {} bpm".format(curr_beat+1, len(rpeaks_all), freq))
            
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
        
        #Prec beat
        elif prec_beat_n_clicks > max_beat_prec_n_clicks:

            print("prec beat")
            max_beat_prec_n_clicks = prec_beat_n_clicks
            if curr_beat is None:
                curr_beat = 0
            else:
                curr_beat -= 1
            freq = compute_frequency(curr_beat)
            curr_beat_label = 'Current Beat: {}/{}. Frequency: {} bpm.'.format(curr_beat+1, len(rpeaks_all), freq)
            #extract prec beat here
            if curr_beat-1 > 0:
                prec_rpeak = rpeaks_all[curr_beat-1]
                xmin = prec_rpeak/fs
            else:
                xmin = 0
            if curr_beat+1 < len(rpeaks_all):
                next_rpeak = rpeaks_all[curr_beat+1]
                xmax = next_rpeak/fs
            else:
                xmax = time[-1]
            
            if xmin < 0:
                xmin = 0    
            if xmax > time[-1]:
                xmax = time[-1]
            fig_global = set_x_range(fig_global, [xmin, xmax])
            change_figure_prec_beat = True

            return json.dumps(""), True, empty_figure(), json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 

        #Next beat
        elif next_beat_n_clicks > max_beat_next_n_clicks:
            
            print("next beat")
            max_beat_next_n_clicks = next_beat_n_clicks
            if curr_beat is None:
                curr_beat = 0
            else:
                curr_beat += 1
            freq = compute_frequency(curr_beat)
            curr_beat_label = 'Current Beat: {}/{}. Frequency: {} bpm.'.format(curr_beat+1, len(rpeaks_all), freq)
            #extract prec beat here
            if curr_beat-1 > 0:
                prec_rpeak = rpeaks_all[curr_beat-1]
                xmin = prec_rpeak/fs
            else:
                xmin = 0
            if curr_beat+1 < len(rpeaks_all):
                next_rpeak = rpeaks_all[curr_beat+1]
                xmax = next_rpeak/fs
            else:
                xmax = time[-1]

            if xmin < 0:
                xmin = 0    
            if xmax > time[-1]:
                xmax = time[-1]

            fig_global = set_x_range(fig_global, [xmin, xmax])
            change_figure_next_beat = True

            return json.dumps(""), True, empty_figure(), json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 

        #Single channel "Ch X" button clicked 
        elif fig_button_handler is not None:

            print("One single channel button clicked: ", curr_chan)
            fig_global = fig_button_handler 
            
            if curr_chan > 0 and curr_chan <= 15:
                if len(anomalies_idx)>0:
                    for i, recon_error in enumerate(reconstruction_errors):
                        if recon_error >= curr_threshold:
                            full_window = 1500
                            half_window = full_window/2
                            x_anomaly = int(half_window + i*full_window)
                            anomalies_idx.append(x_anomaly)
                            fig_global = annotate("Reconstruction Error: {}".format(round(recon_error, 5)), x_anomaly, True)[0] 
                anomaly_id = 0
                text_error_label_to_return, hidden, fig_global = manual_update_plot(None, None, anomaly_id)
            change_figure_button_clicked = True 
            if ind is not None:
                fig_clicked = color_clicked(ind)
            else: 
                fig_clicked = None
            return json.dumps(""), True, empty_figure(), json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 

        #Save HTML
        elif save_n_clicks > max_save_n_clicks:

            print("save plot")
            max_save_n_clicks = save_n_clicks
            save_html()
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config

        #Save text annotations
        elif export_n_clicks > max_export_n_clicks:
            
            print("export annotations")
            max_export_n_clicks = export_n_clicks
            export_annotations()
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config

        #When the user use the dash slider to set x_range
        elif changed_x_range:
            if ind is None:
                print("changed x_range to [{}, {}]".format(curr_xmin, curr_xmax))
                fig_global = set_x_range(fig_global, x_range)
                return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config
            else:
                fig_clicked = set_x_range(fig_clicked, x_range)
                return json.dumps(""), True, fig_clicked, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config 
        else:

            print("no update")
            text_anomaly_label = json.dumps("Threshold must be an float number between {} and {}".format(round(min(reconstruction_errors), 5), round(max(reconstruction_errors), 5)))
            return json.dumps(""), True, fig_global, json.dumps([ind, annotation]), text_anomaly_label, curr_beat_label, config

class ECG_GUI():
    """
    Graphic User Interface Object for ECG Anomaly Detection Auto Encoder.
    """

    def __init__(self, master):
        """
        Initialize all attributes of the object
        """

        self.bg = "black"

        self.path_sep = os.sep
        self.gui_images_path = os.getcwd()+ self.path_sep+"gui_images"+self.path_sep
        
        #update plot 
        self.plot_html_name = None
        self.html_plot_path = "http://127.0.0.1:8050/"

        #Initialize and set some parameters for the GUI window
        self.window = master
        self.window.title("ECG Anomaly Detector AutoEncoder")
        self.window.configure(bg='black')   
        self.window.rowconfigure(10, minsize = 50)
        self.window.columnconfigure(6, minsize = 50)
                
        #name of the program to display always on the top of the window
        self.label_program_name = tk.Label(self.window, text = 'ECG Anomaly Detector', fg = "white", bg = self.bg, font = (None, 30), height = 2)
        self.label_program_name.grid(row=0, column=2, columnspan=2) 
        
        #self.window.iconbitmap("{}vqa.ico".format(self.gui_images_path)) #icon, works only in Windows
        #icon windows and linux tested
        vqa_icon = tk.PhotoImage(file="{}vqa.png".format(self.gui_images_path))
        self.window.tk.call("wm", "iconphoto", self.window._w, vqa_icon)

        img_unicz = tk.PhotoImage(file="{}unicz.png".format(self.gui_images_path))      
        self.unicz_image_label = tk.Label(self.window, image = img_unicz)
        self.unicz_image_label.image = img_unicz
        self.unicz_image_label.grid(row=0, column=0)
        img_unical = tk.PhotoImage(file="{}unical.png".format(self.gui_images_path))      
        self.unical_image_label = tk.Label(self.window, image = img_unical)
        self.unical_image_label.image = img_unical
        self.unical_image_label.grid(row=0, column=5)

        #welcome message
        self.welcome = tk.Label(self.window, text=" ECG Anomaly Detector AutoEncoder \n Software Available under CC-BY License \n Free for Academic Usage \n \n University of Calabria \n Magna Graecia University of Catanzaro \n",
                                foreground="white", bg = self.bg)
        self.welcome.grid(row=2, column=2, columnspan=2)

        #initialize frames
        self.parameters_fr = tk.Frame(self.window, bg = self.bg)                        #for parameters 
        self.load_fr = tk.Frame(self.window, bg = self.bg)                              #for load ecg
        self.anomaly_detection_run_fr = tk.Frame(self.window, bg = self.bg)             #for anomaly detection run   

        #model 
        
        if torch.cuda.is_available():
            self.model = dpnet_loader.load() #load auto encoder model
            self.device = torch.device("cuda:0")
        else:
            self.model = dpnet_loader.load_cpu()
            self.device = torch.device("cpu")
        #print(self.model)
        #initialize back and reset button, the first "scene" of course doesn't need a back or reset button
        self.reset_button = None

        #upload label
        self.upload_label = tk.Label(self.load_fr, text='Upload a .data or .csv ECG file with fs = 500 and 15 channels', bg = "gray", fg = "white")

        self.browser = None 

        self.loadECGScene()

    def load_data(self):
        """
        Load ECG data. 
        Format Supported:
        * .data
        * .csv

        ECG data must be:
        * sampling frequency 500 Hz
        * number of channels 15            
        """
        global signal 
        global annotations 
        global text_error_label
        global nchs 
        global filename 
        global curr_xmax
        global fs 
        global curr_threshold
        
        file_path = askopenfile(mode='r', filetypes=[('ECG Data Files', ['*data', '*csv', '*dat'])]) 
        if file_path is None:
            self.loadECGScene()
        else:
            self.upload_label['text'] = file_path.name
        
            if file_path is not None:
                extension = os.path.splitext(file_path.name)[1]
                basename = os.path.basename(file_path.name)
            
                if extension == ".data":
                    filename = basename[:len(basename)-5]
                    signal = load_object(basename, os.path.dirname(file_path.name))
                    nchs_data, n_record = signal.shape
                    curr_xmax = n_record/fs
                    annotations = [""]*n_record
                    curr_threshold = 0.0063
                    self.loadECGScene()
                elif extension == ".csv": 
                    filename = basename[:len(basename)-4]
                    with open(file_path.name, 'r') as csvdata:
                        sniffer = csv.Sniffer()
                        dialect = sniffer.sniff(csvdata.readline())
                        delimiter = dialect.delimiter
                    signal = np.genfromtxt(file_path.name, delimiter = delimiter, dtype = np.float32)
                    nchs_data, n_record = signal.shape
                    curr_xmax = n_record/fs
                    annotations = [""]*n_record
                    curr_threshold = 0.0063
                    self.loadECGScene()
                elif extension == ".dat":
                    path_record = file_path.name
                    signal, fields = wfdb.rdsamp(path_record[:-4])
                    signal = multichannel_resample(signal, int(signal.shape[0]/2)) #from 1000hz to 500hz
                    signal = myfilter(0.5, 100, 50, signal)
                    nchs_data, n_record = signal.shape
                    curr_xmax = n_record/fs
                    annotations = [""]*n_record
                    curr_threshold = 0.0027 #use threshold for real data
                    self.model.load_state_dict(torch.load("model.pt"))
                    self.model = self.model.to(self.device) #use model trained with real data
                    self.loadECGScene()

                if nchs_data != nchs:
                    if n_record == 15:
                        print("Transposing the given signal.")
                        signal = signal.T
                    else:
                        signal = None
                        text_error_label += "This model is trained using 15 channel ECG 500Hz data. Given a {} channel ECG data.".format(nchs_data)
            
                self.plot_html_name = "ECG_plot_"+filename+".html"

            else:
                text_error_label += "Please Upload a valid .data file before click the Start Button."
                self.loadECGScene()

            self.error_label = tk.Label(self.load_fr, text = text_error_label, bg = self.bg, fg = "red")
            self.error_label.grid(row=4, column=2, columnspan=2, padx=50)
            text_error_label = ""

    def loadECGScene(self):

        """
        GUI Load Data Initial Scene
        """
        global signal 
            
        self.load_fr.grid(row=3, column=2)
        
        self.upload_label.grid(row=3, column=2)

        photo_choose = (tk.PhotoImage(file = r"{}button_choose-file.png".format(self.gui_images_path))).subsample(2,2)
        choose_button = tk.Button(
                            self.load_fr,
                            text="Choose File",
                            image = photo_choose,
                            bg = self.bg,
                            command = self.load_data
                            )
        choose_button.image = photo_choose
        choose_button.grid(row=3, column=3)

        photo_start = (tk.PhotoImage(file = r"{}button_start.png".format(self.gui_images_path))).subsample(2,2)
        start_button = tk.Button(
                            self.load_fr,
                            text="Start Anomaly Detection",
                            image = photo_start,
                            bg = self.bg,
                            command = self.anomaly_detection_run
                            )
        start_button.image = photo_start
        start_button.grid(row=5, column=2, columnspan=2, padx = 50)

        img_vqa = tk.PhotoImage(file="{}vqa.png".format(self.gui_images_path))      
        vqa_image_label = tk.Label(self.load_fr, image = img_vqa)
        vqa_image_label.image = img_vqa
        vqa_image_label.grid(row=6, column=2, columnspan=2, padx = 50) 
    
    def anomaly_detection_run(self):
        
        """
        GUI Run Scene: anomaly detection and rpeaks detection with progress bar.
        """
        global signal 
        global reconstruction_all
        global reconstruction_error
        global fig_global
        global rpeaks_all 
        global isreset 

        self.previous_scene = "LoadECG"
        self.load_fr.grid_remove()
        if isreset:
            isreset = False
            
        if signal is None:
            self.loadECGScene()
        else:
            nchs, n_record = signal.shape
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            windows = sliding_window(signal, 2000, 1500)
            windows = torch.from_numpy(np.array(windows))
            reconstruction_all = np.zeros((nchs, n_record))
        
            interval = 1500
            interval2 = 2000
            n_win = windows.shape[0]
        
            #do anomaly detection with progress bar
            self.anomaly_detection_run_fr.grid(row = 3, column = 2)
            pb = ttk.Progressbar(self.anomaly_detection_run_fr, orient="horizontal", mode="determinate", length=100)
            pb.grid(row = 3, column = 2, padx=(80, 10))
            pb["value"] = 0.00
            label = tk.Label(self.anomaly_detection_run_fr, text="Auto Encoder Reconstruction... Current progress: {}%".format(pb["value"]), bg = self.bg, fg = "white")
            label.grid(row = 4, column = 2, padx=(100, 30))

            #reset button
            photo_reset = (tk.PhotoImage(file = r"{}button_reset.png".format(self.gui_images_path))).subsample(2,2)
            self.reset_button = tk.Button(self.anomaly_detection_run_fr, text='Reset', command = self.reset, bg = self.bg, image = photo_reset)
            self.reset_button.image = photo_reset
            self.reset_button.grid(row = 5, column = 2, padx=(100, 10))     
        
            img_vqa = tk.PhotoImage(file="{}vqa.png".format(self.gui_images_path))      
            vqa_image_label = tk.Label(self.anomaly_detection_run_fr, image = img_vqa)
            vqa_image_label.image = img_vqa
            vqa_image_label.grid(row=6, column=2,  padx=(100, 10)) 

            self.window.update()

            windows_batch_data = windows.clone().detach().unsqueeze(dim=0x1).to(device)

            for i, window in enumerate(windows):

                if not isreset:

                    if i == 0:
                        start = 0
                        end = interval
                    else:
                        start += int(interval)
                        end += int(interval) 
        
                    batch_data = windows_batch_data[i, 0, :, :].reshape((1, 1, 15, interval2)).to(device, dtype = torch.float)
                    print(start, end)
                    with torch.no_grad():

                        reconstruction = self.model(batch_data)
                        reconstruction_errors.append((F.mse_loss(reconstruction[0, 0, :, :interval], batch_data[0, 0, :, :interval])).item())
                        recon_np = (reconstruction.cpu().detach().numpy()).reshape(1, nchs, interval2)

                        if end > n_record:

                            #reconstruction_all[:, start:n_record] = recon_np[0, :, :interval2-(end-n_record)] 
                            pb["value"] = 100.00
                            label['text'] = "Auto Encoder Reconstruction Complete: {}%".format(pb["value"])
                            pb.grid(row = 3, column = 2, padx=(80, 10))
                            label.grid(row = 4, column = 2, padx=(100, 30))
                            self.window.update()

                        else:

                            reconstruction_all[:, start:end] = recon_np[0, :, :interval]
                            pb["value"] = round(((i + 1) / n_win) * 100, 2)
                            label['text'] = "Auto Encoder Reconstruction... Current progress: {}%".format(pb["value"])
                            pb.grid(row = 3, column = 2, padx=(80, 10))
                            label.grid(row = 4, column = 2, padx=(100, 30))
                            self.window.update()
                else:
                    break

                torch.cuda.empty_cache()
           
            if not isreset:
                
                pb["value"] = 100.00
                label['text'] = "Auto Encoder Reconstruction Complete: {}%".format(pb["value"])
                pb.grid(row = 3, column = 2, padx=(80, 10))
                label.grid(row = 4, column = 2, padx=(100, 30))
                self.window.update()

                #create figure
                fig_global = create_fig()
                pb.grid_remove()
                label.grid_remove()
                self.anomaly_detection_run_fr.grid_remove()
                self.anomaly_detection_run_fr.grid(row = 3, column = 2)

                #find rpeaks and autoannotate them
                pb = ttk.Progressbar(self.anomaly_detection_run_fr, orient="horizontal", mode="determinate", length=100)
                pb.grid(row = 3, column = 2, padx=(80, 10))
                pb["value"] = 0.00
                label = tk.Label(self.anomaly_detection_run_fr, text="Finding Rpeaks... Current progress {}%".format(pb["value"]), bg = self.bg, fg = "white")
                label.grid(row = 4, column = 2, padx=(100, 30))

                photo_reset = (tk.PhotoImage(file = r"{}button_reset.png".format(self.gui_images_path))).subsample(2,2)
                self.reset_button = tk.Button(self.anomaly_detection_run_fr, text='Reset', command = self.reset, bg = self.bg, image = photo_reset)
                self.reset_button.image = photo_reset
                self.reset_button.grid(row = 5, column = 2, padx=(100, 10))     
                img_vqa = tk.PhotoImage(file="{}vqa.png".format(self.gui_images_path))      
                vqa_image_label = tk.Label(self.anomaly_detection_run_fr, image = img_vqa)
                vqa_image_label.image = img_vqa
                vqa_image_label.grid(row=6, column=2,  padx=(100, 10)) 
                self.window.update()
            
                rpeaks_all = find_rpeaks(signal)
                annotate_rpeaks(self.window, pb, label)

                if not isreset:

                    pb["value"] = 100.00
                    label['text'] = "Finding Rpeaks Complete: {}%".format(pb["value"])
                    pb.grid(row = 3, column = 2, padx=(80, 10))
                    label.grid(row = 4, column = 2, padx=(100, 30))
                    self.window.update()

                    self.plot_ecg_and_anomalies()

                else:
                    self.loadECGScene()
            else:
                self.loadECGScene()

    def replot(self):
        """
        Replot
        """
        thread_cef = Thread(target=self.cef_plot_thread, deamon = True)
        thread_cef.start()

    def plot_ecg_and_anomalies(self):

        """
        Create figures, create and run the dash app 
        """
        global fig_global
        global text_error_label
        global app_dash
        global signal 
        global curr_chan
        global anomalies_idx
        global anomaly_id
        global curr_threshold 
        global reconstruction_errors
        global config 
        global nchs 

        if signal is not None:
            nchs, n_record = signal.shape

        self.previous_scene = "LoadECG"

        self.anomaly_detection_run_fr.grid_remove()
        self.window.update()
        
        """
        self.parameters_fr.grid(row = 3, column=2)     

        photo_replot = (tk.PhotoImage(file = r"{}button_replot.png".format(self.gui_images_path))).subsample(2,2)
        replot_button = tk.Button(
                            self.parameters_fr,
                            text="Replot",
                            image = photo_replot,
                            bg = self.bg,
                            command = self.replot
                            )
        replot_button.image = photo_replot
        replot_button.grid(row=3, column=2, padx=(130, 10))

        #reset button
        photo_reset = (tk.PhotoImage(file = r"{}button_reset.png".format(self.gui_images_path))).subsample(2,2)
        self.reset_button = tk.Button(self.parameters_fr, text='Reset', command = self.reset, bg = self.bg, image = photo_reset)
        self.reset_button.image = photo_reset
        self.reset_button.grid(row=4, column=2, padx=(130, 10))
              
        img_vqa = tk.PhotoImage(file="{}vqa.png".format(self.gui_images_path))      
        vqa_image_label = tk.Label(self.parameters_fr, image = img_vqa)
        vqa_image_label.image = img_vqa
        vqa_image_label.grid(row=5, column=2, padx=(130, 10)) 
        self.window.update()

        """

        app_dash_layout_args = [
            
            dcc.Input(
                    id="annotation",
                    type="text",
                    placeholder="Text Annotation Point",
                    style={"position": "absolute", "left": "50px", "top": "10px", 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            html.Label(
                    "Clicked, Annotation",
                    id="clicked-annotation-label",
                    style={"position": "absolute", "left": "50px", "top": "30px", 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            dcc.Input(
                    id="xmin",
                    type="text",
                    placeholder="Slider x_min",
                    style={"position": "absolute", "left": "250px", "top": "10px", 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            dcc.Input(
                    id="xmax",
                    type="text",
                    placeholder="Slider x_max",
                    style={"position": "absolute", "left": "350px", "top": "10px", 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            dcc.Input(
                    id="anomaly",
                    type="text",
                    placeholder="Anomaly id",
                    style={"position": "absolute", "left": "450px", "top": "10px", 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            html.Label(
                    text_error_label,
                    id="error",
                    hidden=True,
                    style={"color": "red", "position": "absolute", "left": "250px", "top": "30px", 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            html.Button(
                    "<< Go Prec Anomaly",
                    id = "prec-button",  
                    n_clicks=0, 
                    style={"position": "absolute", "right": "200px", "top": "10px",'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "100px", 'height': '40px'}
            ),
            html.Button(
                    ">> Go Next Anomaly",
                    id = "next-button",  
                    n_clicks=0,
                    style={"position": "absolute", "right": "50px", "top": "10px", 'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "100px", 'height': '40px'}
            ),
            dcc.Input(
                    id="threshold",
                    type="text",
                    placeholder=curr_threshold,
                    style={"position": "absolute", "right": "600px", "top": "10px", 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            dcc.Loading(
                dcc.Graph( 
                        id="fig",
                        figure=fig_global,
                        animate=True,
                        style={"position": "absolute", "left": "75px", "top": "70px", 'width': '180vh', 'height': '90vh', 
                               'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'},
                        config = {'modeBarButtonsToAdd': ['drawline', 'drawrect', 'eraseshape'], 'responsive': True, 'displayModeBar': True}

                ),
                id="loading-graph",
                style={"position": "absolute", "left": "600px", "top": "350px", 'width': '400px', 'height': '200px'},
            ),
            html.Button(
                    "Compute Anomalies",
                    id = "anomalies-button",  
                    n_clicks=0,
                    style={"position": "absolute", "right": "625px", "top": "50px", 'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "100px", 'height': '40px'}
            ),
            html.Div(
                dcc.RangeSlider(
                    id='rangeslider',
                    min=0,
                    max=time[-1],
                    value=[0, time[-1]],
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                style={"position": "absolute", "left": "100px", "bottom": "10px", 'width': '150vh', 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            html.Button(
                "Manual Update",
                id = "manual-update-button",  
                n_clicks=0,
                style={"position": "absolute", "left": "350px", "top": "50px", 'cursor': 'pointer', 'border': '0px', 
                       'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '10px',
                       'font-family': 'Open Sans', 'width': "100px", 'height': '30px'}
            ),
            html.Button(
                    "All Channels",
                    id = "allchannels-button",  
                    n_clicks=0,
                    style={"position": "absolute", "left": "3px", "top": "120px", 'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "100px", 'height': '30px'},              
            ),
            html.Button(
                    "Annotate point",
                    id = "annotate-button",  
                    n_clicks=0,
                    style={"position": "absolute", "left": "50px", "top": "50px", 'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "100px", 'height': '30px'}
            ),
            html.Button(
                    "Save",
                    id = "save-button",  
                    n_clicks=0,
                    style={"position": "absolute", "right": "50px", "bottom": "60px", 'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "100px", 'height': '30px'}
            ),
            html.Button(
                    "Export annotations",
                    id = "export-button",  
                    n_clicks=0,
                    style={"position": "absolute", "right": "50px", "bottom": "20px", 'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "100px", 'height': '30px'}
            ),
            html.Label(
                'Show reconstruction error? (can be slow)',
                id='reconerror-label',
                style={"position": "absolute", "right": "250px", "top": "100px", 'width': '200px', 'height': '50px', 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            dcc.RadioItems(
                ['Yes', 'No'],
                'No',
                id='show-recon-radio',
                inline=True,
                style={"position": "absolute", "right": "200px", 'width': '50px', 'height': '20px', "top": "100px", 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            html.Label(
                    "Choose a threshold between {} and {}".format(round(min(reconstruction_errors), 5), round(max(reconstruction_errors), 5)),
                    id="anomaly-label",
                    style={"position": "absolute", "right": "450px", "top": "30px", 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            html.Button(
                    "Reload",
                    id = "reload-button",  
                    n_clicks=0,
                    style={"position": "absolute", "right": "50px", "top": "100px", 'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "100px", 'height': '30px'}
            ),
            html.Label(
                'Visualize next/prec beat',
                id='beat-label',
                style={"position": "absolute", "right": "50px", "bottom": "150px", 'width': "100px", 'height': '60px', 'backgroundColor':'rgb(17, 17, 17)', 'color': 'white'}
            ),
            html.Button(
                    "Next",
                    id = "beatnext-button",  
                    n_clicks=0,
                    style={"position": "absolute", "right": "50px", "bottom": "100px", 'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "50px", 'height': '30px'}
            ),
            html.Button(
                    "Prec",
                    id = "beatprec-button",  
                    n_clicks=0,
                    style={"position": "absolute", "right": "100px", "bottom": "100px", 'cursor': 'pointer', 'border': '0px', 
                           'border-radius': '3px', 'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                           'font-family': 'Open Sans', 'width': "50px", 'height': '30px'}
            )
        ]
        for i in range(nchs):
            ch_name = channels_names[i]
            single_chX_button = html.Button(
                        ch_name,
                        id = "channel{}-button".format(i+1),  
                        n_clicks=0,
                        style={"position": "absolute", "left": "3px", "top": "{}px".format(120+((i+1)*40)),
                               'cursor': 'pointer', 'border': '0px', 'border-radius': '3px',
                               'background-color': 'rgb(31, 24, 252)', 'color': 'white', 'font-size': '12px',
                                'font-family': 'Open Sans', 'width': "100px", 'height': '33px'}
            )
            app_dash_layout_args.append(single_chX_button)

        app_dash.layout = html.Div(
            app_dash_layout_args,
            style = {'border': '0px', 'backgroundColor':'rgb(17, 17, 17)', 'background-size': '100%', 'position': 'fixed',
                           'width': '100%', 'height': '100%'}
        )

        #self.plot_thread = Thread(target=self.cef_plot_thread, daemon=True)
        #self.plot_thread.start()

        webbrowser.open(self.html_plot_path)
        
        app_dash.run_server(debug=False)#threaded = True
        
        self.reset()
        self.loadECGScene()
        
    def reset(self):

        """
        Reset
        """
        global max_manual_n_clicks
        global max_annotate_n_clicks
        global max_prec_n_clicks 
        global max_next_n_clicks 
        global max_save_n_clicks 
        global max_anomalies_n_clicks 
        global max_export_n_clicks 
        global max_reload_n_clicks
        global max_beat_prec_n_clicks 
        global max_beat_next_n_clicks 

        global text_error_label 
        global anomalies_idx 
        global anomaly_id 
        global curr_threshold 
        global reconstruction_all 
        global reconstruction_errors 
        global annotations 
        global annotated_points 
        global fig_global
        global signal 
        global nchs
        global curr_chan               #from 1 to 15, 0 for "all channels"
        global curr_xmin               #current xmin slider value
        global curr_xmax               #current xmax slider value
        global curr_beat
        global colors 
        global filename 
        global max_channel_n_clicks
        global change_figure_button_clicked 
        global change_figure_anomalies_computed  
        global change_figure_annotation 
        global change_figure_prec_beat 
        global change_figure_next_beat 
        global change_reload
        global fig_clicked 
        global prec_clicked 
        global prec_show_recon_error 
        global show_recon_error 
        global fig_all 
        global fig_single_ch
        global time 
        global fs   #model is trained with ECG data 500Hz frequency sampling
        global rpeaks_all 
        global prec_mode 
        global config 
        global fig_global
        global fig_all
        global fig_single_ch
        global isreset 

        """
        Re-initialize all the frames and parameters selected.
        """
        print("resetting")
        isreset = True

        self.parameters_fr.grid_remove()
        self.load_fr.grid_remove()
        self.anomaly_detection_run_fr.grid_remove()
        self.upload_label.grid_remove()
        self.parameters_fr = tk.Frame(self.window, bg = self.bg)                        #for parameters 
        self.load_fr = tk.Frame(self.window, bg = self.bg)                              #for load ecg
        self.anomaly_detection_run_fr = tk.Frame(self.window, bg = self.bg)             #for anomaly detection run   
        self.upload_label = tk.Label(self.load_fr, text='Upload a .data, .csv or .dat ECG file with fs = 500 and 15 channels', bg = "gray", fg = "white")

        max_manual_n_clicks = 0
        max_annotate_n_clicks = 0
        max_prec_n_clicks = 0
        max_next_n_clicks = 0
        max_save_n_clicks = 0
        max_anomalies_n_clicks = 0
        max_export_n_clicks = 0
        max_reload_n_clicks = 0
        max_beat_prec_n_clicks = 0
        max_beat_next_n_clicks = 0

        text_error_label = ""
        anomalies_idx = []
        anomaly_id = None
        curr_threshold = None
        reconstruction_all = None
        reconstruction_errors = []
        annotations = None
        annotated_points = []
        fig_global = None
        signal = None 
        nchs = 15
        curr_chan = 0 #from 1 to 15, 0 for "all channels"
        curr_xmin = 0                  #current xmin slider value
        curr_xmax = None               #current xmax slider value
        curr_beat = None
        colors = ["green", "white", "blue", "lightcoral", "orange", "cyan", "magenta", "yellow", "purple", "hotpink", "midnightblue", "lime", "olive", "gold", "palegreen"]
        filename = "ECG_file"
        max_channel_n_clicks = [0]*(nchs+1)
        change_figure_button_clicked = False 
        change_figure_anomalies_computed = False 
        change_figure_annotation = False 
        change_figure_prec_beat = False
        change_figure_next_beat = False
        change_reload = False
        fig_clicked = None 
        prec_clicked = None 
        prec_show_recon_error = "No"
        show_recon_error = "No"
        fig_all = None
        fig_single_ch = [None]*nchs
        time = None 
        fs = 500   #model is trained with ECG data 500Hz frequency sampling
        rpeaks_all = None 
        prec_mode = "Select"
        config = {"modeBarButtonsToAdd": []}

        fig_all = None
        fig_single_ch = [None]*nchs

        self.loadECGScene()

#BUG LIST
#0. ERROR WHEN PRESSING THE BUTTON "NEXT ANOMALY" BUT THE USER IS VISUALIZING THE LAST ANOMALY 
#1. ERROR WHEN PRESSING THE BUTTON "PREC ANOMALY" BUT THE USER IS VISUALIZING THE FIRST ANOMALY 

#MAYBE ADD
#0. SAVE SHAPE ANNOTATIONS WHEN SWITCHING IN SELECT MODE
                  
#TO DO    
#10. Delete active text annotation button
