# README #


### Requirements ###
Cuda must be available.

    pip3 install dill
    pip3 install torch

### Data & Model ###
You can download [data](https://mega.nz/file/ilB21aYA#b4tolz6ff03JFPzCoR_Ls5XgY6d1cNZCTfoGLUpixEo) and the [model](https://mega.nz/file/ztJjFLoQ#5YAweUyFRonGHY1J7sM920s6qAZ7mnWH4X5Jp6PRmOQ) clicking from here.

### Loading data ###
    from pickleio.PickleIO import load_object
    data = load_object([file], [directory])

### Splitting Data in Windows ###

    from windows.WindowingUtils import sliding_window
    windows = sliding_window(data, int(0x5 * 0.8 * 0x1F4), stride=int(0.2 * 0x1F4))

### Loading Model ###

Edit **conf.py** specifying model's path to use ***load()*** with no file path.

    import dpnet.dpnetloader as dpnet_loader
    dpNet = dpnloader.load()

### Getting Reconstruction Error ###

    import torch.nn.functional as F
    batch_data = None # define your own
    decoded = dpNet(batch_data)
    reconstruction_errors = F.mse_loss(decoded, batch_data)
