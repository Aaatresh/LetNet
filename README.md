# LetNet
Implemention of LetNet architecture framework. A time unfolded network for image reconstruction on diffusercam.

Here, the image reconstruction process of an image captured using diffusercam has been time unfolded into a neural netowrk to learn the mappings between sensor readings and their corresponding ground turth images. In place of a soft thresholding function at each time step $t$, we use the linear expansion of thresholds(LET) for a more efficient and robust performance.

# Data
The data can be obtained using a diffusercam by capturing the raw sensor readings after calibration. Sensor readings and their ground truths form input,output pairs for training.

# Running the code
Once the dataset has been created correctly, LetNet can be trained as:
```
python3 letnet_fixed_multilayer.py
```
