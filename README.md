# Superresolution-of-Bathymetric-Data
This repository contains the code and neural networks that have been used to gain the results in my Master's thesis paper "Super-Resolution of Bathymetric Data Using Diffusion Models". The paper can be found Here:xxxxx

This work is also a direct extension of J.Shims et.al work so alot of the code is taken from their amazing work. Here you can find their [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Shim_Diffusion-Based_Signed_Distance_Fields_for_3D_Shape_Generation_CVPR_2023_paper.html) and their [repository](https://github.com/Kitsunetic/SDF-Diffusion?tab=readme-ov-file).

##Training
To run the training the command is:
```sh
python3 main.py config/sr32_64/sonar.yaml --gpus 0,1,2,....
```
where if 0 is the first gpu, 1 is the second and so on. You can use as many gpus as you want.
sonar.yaml is the configuration file. There are some variables there that are more important than others which are explained below.

Important variables New Loss Model (NLM) (explained in the paper):
- new_loss: There are two places where this variable occurs; under the ddpm and the preprocessor. BOTH has to be set to "yes" for the model to be trained with the modified loss explained in the paper.
- sdf_weight: This variable is set as gamma in the paper and it decides how much the model should focus on having correct distances between two adjacent cells in the outputet SDF

Important variables for models using intensity:
in_channel: This is the number of input channels in the network. By default, this is 2. However, we want to add the intensity to another channel so it needs to be set to 3.
use_intensity: set this to "yes" to train the model with intensity data

##Testing
To test the networks performance and compare it against an interpolation technique, you can run the networkvsinterp.py script. 
You can adjust variables in the script that are marked such as what interpolation technique to compare against. An important thing to note is that the configuration file (sonar.yaml) has to be configured for the model you are testing.
For example, the new_loss variables have to be "yes" if you test a NLM and in_channel and useintensity has to be set to "yes" if you're testing a model that has been trained with intensity values.








