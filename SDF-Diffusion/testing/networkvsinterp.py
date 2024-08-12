print("Hej!")
from pathlib import Path

import torch as th
import torch.nn.functional as F
import numpy as np
import yaml
from easydict import EasyDict
from glob import glob


import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from src.utils import instantiate_from_config
import pdb
import cv2





path = Path("../data/sonar_data")


# network_path = "../results/sr32_64/OLM/best_ep0983.pth"  #vanilla         
# network_path = "../results/sr32_64/NLM/best_ep0983.pth"  #newloss   	   
network_path = "../results/sr32_64/OLMI/best_ep0984.pth" #intensity        
# network_path = "../results/sr32_64/NLMI/best_ep0988.pth"  #newloss and intensity 





plot = False # if true, will plot the output                                                        |
use_intensity = True # if true, the model uses intensity data                                       |
new_loss_model = False #if true, the model uses the new loss function                               |
loops = 10 #number of times to sample from the network per input and then average the output        |   THESE ARE THE PARAMETERS YOU CAN CHANGE 
cascade = False # True = 32 -> 64 -> 128, False = 32 -> 64                                          |
interpolation_method = "bicubic" #bicubic or bilinear or pyrUp                                      |             
split = "test" #Pick the split you want to test on (train, val, test)                               |   



print("Network path: ", network_path)
sr64_args_path = "../config/sr32_64/sonar.yaml"
with open(sr64_args_path) as f:
    args2 = EasyDict(yaml.safe_load(f))

#Extract the model, sampler and preprocessor from the config YAML
model2 = instantiate_from_config(args2.model).cuda()
ckpt = th.load(network_path, map_location="cpu")
model2.load_state_dict(ckpt["model"])
print(ckpt["best_loss"])
ddpm_sampler2 = instantiate_from_config(args2.ddpm.valid).cuda()
preprocess = instantiate_from_config(args2.preprocessor, "cuda")


# Load the data
directories = sorted(glob("../../data/sonar_data/*/", recursive=True))
filelist = []

for directory in directories:
    path = Path(directory)
    files = sorted(path.glob("*.npz"))
    for file in files:
        if np.load(file)["data_split"] == split:
            filelist.append(file)
    
print("Length of data: ", len(filelist))



interplossl1 = []
networklossl1 = []
interplossl2 = []
networklossl2 = []
interploss_PSNR = []
networkloss_PSNR = []
iteration = 1
filtervalue = -2.5
mean_distanceloss = []
mean_variance= []
if cascade:
    res = 128
else:
    res = 64
print("Averaging over " + str(loops) + " loops for the network output")


heightlist = []
for filen in filelist:
    print("iteration: " + str(iteration) + "/" + str(len(filelist)))
    iteration += 1
    data = np.load(filen)
    
    center = data["center"] #used later for plotting the maps on the correct location
    sdf = data["sdf"]

    filtered_sdf = th.from_numpy(sdf).float().unsqueeze(0).unsqueeze(0).cuda()
    filtered_sdf[filtered_sdf != filtered_sdf] = float('inf')  #remove nans

    standardized_sdf = preprocess.standardize(filtered_sdf, 1) #Preprocess the sdf, this will just be used to check the distance loss of the network output (check if the distance between two adjacent cells is 2/63)
    
    heightmap_truth = -sdf[:,:,(res-1)] #extract the heightmap from the sdf
    
    non_nan_indices = np.where(heightmap_truth == heightmap_truth) #find the indices of the non-nan values
    meterres = data["hrmeterres"]
    if cascade:
        lrmeterres = data["lrmeterres2"]        #this has to do with how the data is saved, when using 128x128x128 data, the lowest meterresolution is saved as lrmeterres2
    else:
        lrmeterres = data["lrmeterres"]
    
    print(meterres)
    mask_matrix = np.zeros_like(heightmap_truth)
    mask_matrix[non_nan_indices] = 1

    #apply erosion with a 3x3 cross. 
    #This step is to ensure that we remove the prediced values on the border of the heightmap when we later compare the network output and interpolation output to the ground truth
    # This is because interpolation will struggle with the border so we dont want an unfair advantage for the network
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    mask_matrix = cv2.erode(mask_matrix, kernel, iterations=3)
    heightmap_truth[mask_matrix == 0]   = filtervalue
    mask = np.where(heightmap_truth == filtervalue)

    heightmap_truth_PSNR = heightmap_truth.copy()
    heightmap_truth = (heightmap_truth/-2) * -meterres[2]*res 
  
    
    filtervalue_meter = (filtervalue/-2) * -meterres[2]*res
    lrfiltervalue_meter = (filtervalue/-2) * -lrmeterres[2]*32

    

    if cascade:
        sdf_ds = data["sdf_ds2"] #same as above, when using 128x128x128 data, the downsampled sdf is saved as sdf_ds2
    else:
        sdf_ds = data["sdf_ds"]

    
    sdf_ds = th.from_numpy(sdf_ds).float().unsqueeze(0).unsqueeze(0).cuda()
    #change all nans to inf
    sdf_ds[sdf_ds != sdf_ds] = float('inf')
    lr_cond = F.interpolate(sdf_ds, (64, 64, 64), mode="nearest")
    lr_cond = preprocess.standardize(lr_cond, 0)
    if use_intensity:
        sdf_intensity = data["intensity"]
        sdf_intensity_tensor = th.from_numpy(sdf_intensity).float().unsqueeze(0).unsqueeze(0).cuda()
        int_cond = F.interpolate(sdf_intensity_tensor, (64, 64, 64), mode="nearest")

    print("Sampling output from network...")
    heightmap_mean = np.zeros((res,res))
    distance_loss = []
    height_map_list = []
    for i in range(loops):
        if not use_intensity:
            out2 = ddpm_sampler2.sample_ddim(lambda x, t: model2(th.cat([lr_cond, x], 1), t), (1, 1, 64, 64, 64), show_pbar=False)
        else: 
            out2 = ddpm_sampler2.sample_ddim(lambda x, t: model2(th.cat([lr_cond, int_cond, x], 1), t), (1, 1, 64, 64, 64), show_pbar=False)
        if not cascade:
            #check the distance loss of the network output
            rolled_x_0 = th.roll(standardized_sdf, 1, 4)
            rolled_x_0[:, :, :, 0] = 0
            mask2 = (th.abs(standardized_sdf) < 1 ) & (th.abs(rolled_x_0) < 1)
            diff = th.diff(out2, dim=4)
            diff = F.pad(diff, (1, 0))
            diff = diff[mask2]
            diff = diff[diff != 0]
            loss = th.mean(th.abs(th.abs(diff)-1))
            distance_loss.append(loss.item()*(2/(res-1)))
            out2 = preprocess.destandardize(out2, 1)

        
        else:
            #cascade the network output into the network again to gain an even higher resolution. Then check distance loss again
            out2 = preprocess.destandardize(out2, 1)
            out2_cond = F.interpolate(out2, (128, 128, 128), mode="nearest")
            out2_cond = preprocess.standardize(out2_cond, 1)
            out2 = ddpm_sampler2.sample_ddim(lambda x, t: model2(th.cat([out2_cond, x], 1), t), (1, 1, 128, 128, 128), show_pbar=False)
            rolled_x_0 = th.roll(standardized_sdf, 1, 4)
            rolled_x_0[:, :, :, 0] = 0
            mask2 = (th.abs(standardized_sdf) < 1 ) & (th.abs(rolled_x_0) < 1)
            diff = th.diff(out2, dim=4)
            diff = F.pad(diff, (1, 0))
            diff = diff[mask2]
            diff = diff[diff != 0]
            loss = th.mean(th.abs(th.abs(diff)-1))
            distance_loss.append(loss.item()*(2/(res-1)))
            out2 = preprocess.destandardize(out2, 2)
       
        
        out2 = out2.squeeze().cpu().numpy()
        #convert 3d shape to heightmap_out
        heightmap_out = np.ones((res, res))*filtervalue

        
        if new_loss_model:
            clip_value = 2/(res-1)
        else: 
            clip_value = 2/res

        #find the cell with the lowest distance value and set the heightmap_out value to the corresponding value
        for i in range(out2.shape[0]):
            for j in range(out2.shape[1]):
                    line = out2[i,j,:]
                    line = np.abs(line)
                    minind = np.argmin(line)
                    if out2[i, j, minind] !=clip_value:
                        heightmap_out[i, j] = -(out2[i, j, minind] + (len(out2[i,j,:]) - minind-1)*2/(res-1))

        heightmap_out[mask] =filtervalue
        height_map_list.append(heightmap_out)
        heightmap_mean += heightmap_out
    
    #Average the output of the network and calculate the variance. heightmap_psnr contains the normalized heightmap for the PSNR calculation while heightmap_out contains the heightmap in meters
    heightmap_out = heightmap_mean/loops
    heightmap_out_PSNR = heightmap_out.copy()
    heightmap_out = (heightmap_out/-2) * -meterres[2]*res
    height_map_variance = np.zeros((res,res))
    for heightmap in height_map_list:
        heightmap = (heightmap/-2) * -meterres[2]*res
        height_map_variance += (heightmap - heightmap_out)**2
    height_map_variance = height_map_variance/loops

    mean_distance = sum(distance_loss)/len(distance_loss)
    print("SDF distance loss: ", mean_distance)
    mean_distanceloss.append(mean_distance)
    mean_variance.append(np.mean(height_map_variance))
    print("Variance:", mean_variance[-1])
    
  
    
    

    if cascade:
        sdf_ds = data["sdf_ds2"]
    else:
        sdf_ds = data["sdf_ds"]

   
    
    
    heightmap_interp = -sdf_ds[:,:,31]
    nan_indices = np.where(heightmap_interp != heightmap_interp)
    heightmap_interp[nan_indices] = filtervalue
    heightmap_lr = heightmap_interp.copy()
    heightmap_lr =  (heightmap_lr/-2) * -lrmeterres[2]*32

    #Interpolate the heightmap using the specified method
    if interpolation_method != "pyrUp":
        heightmap_interp = th.from_numpy(heightmap_interp).float().unsqueeze(0).unsqueeze(0).cuda()
        if cascade:
            heightmap_interp = F.interpolate(heightmap_interp, (128, 128), mode=interpolation_method)
        else:
            heightmap_interp = F.interpolate(heightmap_interp, (64, 64), mode=interpolation_method)
        heightmap_interp = heightmap_interp.squeeze().cpu().numpy()
    else:
        heightmap_interp = cv2.pyrUp(heightmap_interp)
        if cascade:
            heightmap_interp = cv2.pyrUp(heightmap_interp)
    heightmap_interp[mask] = filtervalue
    heightmap_interp_PSNR = heightmap_interp.copy()
    heightmap_interp = (heightmap_interp/-2) * -meterres[2]*res

        
    
    

    #Calculate the PSNR, L1 and L2 loss of the network output and the interpolation output by comparing it to the ground truth that has the border removed
    mask3 = np.where(heightmap_truth != filtervalue_meter)
    interplossl1.append(F.l1_loss(th.from_numpy(heightmap_interp[mask3]).float().cuda(), th.from_numpy(heightmap_truth[mask3]).float().cuda()))
    networklossl1.append(F.l1_loss(th.from_numpy(heightmap_out[mask3]).float().cuda(), th.from_numpy(heightmap_truth[mask3]).float().cuda()))
    interplossl2.append(F.mse_loss(th.from_numpy(heightmap_interp[mask3]).float().cuda(), th.from_numpy(heightmap_truth[mask3]).float().cuda()))
    networklossl2.append(F.mse_loss(th.from_numpy(heightmap_out[mask3]).float().cuda(), th.from_numpy(heightmap_truth[mask3]).float().cuda()))

    print("mae interp_loss: ", interplossl1[-1], "\nmae Network_loss: ", networklossl1[-1])
    print("mse interp_loss: ", interplossl2[-1], "\nmse Network_loss: ", networklossl2[-1])

    interp_PSNR = F.mse_loss(th.from_numpy(heightmap_interp_PSNR[mask3]).float().cuda(), th.from_numpy(heightmap_truth_PSNR[mask3]).float().cuda())
    network_PSNR = F.mse_loss(th.from_numpy(heightmap_out_PSNR[mask3]).float().cuda(), th.from_numpy(heightmap_truth_PSNR[mask3]).float().cuda())

    interploss_PSNR.append(10*np.log10(4/interp_PSNR.item())) #the formula for PSNR is 10*log10(max^2/MSE) and max value in the heightmap is 2 (since the normalized since the distance from the top of the sdf (1) to the bottom (-1) is 2)
    networkloss_PSNR.append(10*np.log10(4/network_PSNR.item()))


    print("interp_PSNR: ", interploss_PSNR[-1], "\nNetwork_PSNR: ", networkloss_PSNR[-1])


    if plot: 
        #Plot the heightmaps in the correct location
        low_limx = center[0] - (315/2)
        high_limx = center[0] + (315/2)
        low_limy = center[1] - (315/2)
        high_limy = center[1] + (315/2)

        print(low_limx,high_limx,low_limy,high_limy)
        
        plt.figure()
        heightmap_plot = heightmap_lr.copy()
        heightmap_plot[heightmap_lr == lrfiltervalue_meter] = np.nan
        plt.imshow(heightmap_plot, cmap="viridis", vmin=-25, extent = [low_limx,high_limx,low_limy,high_limy])
        plt.title("Low Resolution Map")
        plt.xlabel("Eastings (m)")
        plt.ylabel("Northings (m)")
        plt.colorbar(label = "depth (m)")
        if use_intensity:
            plt.figure()
            heightmap_plot = sdf_intensity[:,:,31]
            heightmap_plot[heightmap_lr == lrfiltervalue_meter] = np.nan
            plt.imshow(heightmap_plot, cmap="plasma", extent = [low_limx,high_limx,low_limy,high_limy])
            plt.title("Intensity Map")
            plt.xlabel("Eastings (m)")
            plt.ylabel("Northings (m)")
            plt.colorbar(label = "Normalized Intensity")



        plt.figure()
        heightmap_plot = heightmap_truth.copy()
        heightmap_plot[heightmap_plot == filtervalue_meter] = np.nan
        plt.imshow(heightmap_plot, cmap="viridis", vmin=-25, extent = [low_limx,high_limx,low_limy,high_limy])
        plt.title("Ground Truth (High Resolution)")
        plt.xlabel("Eastings (m)")
        plt.ylabel("Northings (m)")
        plt.colorbar(label = "depth (m)")

        heightmap_plot = heightmap_out.copy()
        heightmap_plot[heightmap_plot == filtervalue_meter] = np.nan
        plt.figure()
        plt.imshow(heightmap_plot, cmap="viridis", vmin=-25, extent = [low_limx,high_limx,low_limy,high_limy])
        plt.title("Network Output")
        plt.xlabel("Eastings (m)")
        plt.ylabel("Northings (m)")
        plt.colorbar(label = "depth (m)")
        
        heightmap_plot = heightmap_interp.copy()
        heightmap_plot[heightmap_plot == filtervalue_meter] = np.nan
        plt.figure()
        plt.imshow(heightmap_plot, cmap="viridis", vmin=-25, extent = [low_limx,high_limx,low_limy,high_limy])
        plt.title("Bicubic Interpolation")
        plt.xlabel("Eastings (m)")
        plt.ylabel("Northings (m)")
        plt.colorbar(label = "depth (m)")
        

        plt.figure()
        plt.imshow(np.abs(heightmap_out-heightmap_truth), cmap="inferno", vmax=2, extent = [low_limx,high_limx,low_limy,high_limy])
        plt.title("Network Error")
        plt.xlabel("Eastings (m)")
        plt.ylabel("Northings (m)")
        plt.colorbar(label = "error (m)")

        plt.figure()
        plt.imshow(np.abs(heightmap_interp-heightmap_truth), cmap="inferno", vmax=2, extent = [low_limx,high_limx,low_limy,high_limy])
        plt.title("Bicubic Interpolation Error")
        plt.xlabel("Eastings (m)")
        plt.ylabel("Northings (m)")
        plt.colorbar(label = "error (m)")


        plt.figure()
        plt.imshow(np.abs(height_map_variance), cmap="inferno", vmax=2, extent = [low_limx,high_limx,low_limy,high_limy])
        plt.title("Network Variance")
        plt.colorbar()


        plt.show(block=True)

# print mean errors, loss and variance
print("mean interploss l1", sum(interplossl1)/len(interplossl1))
print("mean networkloss l1", sum(networklossl1)/len(networklossl1))
print("mean interploss l2", sum(interplossl2)/len(interplossl2))
print("mean networkloss l2", sum(networklossl2)/len(networklossl2))
print("mean interpolation PSNR", sum(interploss_PSNR)/len(interploss_PSNR))
print("mean network PSNR", sum(networkloss_PSNR)/len(networkloss_PSNR))
print("mean network sdf distance loss", sum(mean_distanceloss)/len(mean_distanceloss))
print("mean network variance (m)", sum(mean_variance)/len(mean_variance))
