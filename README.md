# LetNet
Implemention of LetNet architecture framework. A time unfolded network for image reconstruction on diffusercam.


## Description

The image reconstruction process of an image captured using diffusercam has been time unfolded into a neural network to learn the mappings between sensor readings and their corresponding ground truth images. In place of a soft thresholding function at each time step t, we use the linear expansion of thresholds(LET) for a more efficient and robust performance.

## Getting Started

### Installing
```
  git clone https://github.com/Aaatresh/LetNet
```

### Data
The data can be obtained using a diffusercam by capturing the raw sensor readings after calibration. Sensor readings and their ground truths form input,output pairs for training.

### Running the code
Once the dataset has been created correctly, LetNet can be trained as:
```
python3 letnet_fixed_multilayer.py
```

## Authors

Contributors names and contact info:
* Anirudh Aatresh (aaa.171ec106@nitk.edu.in)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details





