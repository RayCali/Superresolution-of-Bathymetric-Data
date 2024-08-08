
import numpy as np
import os
from pathlib import Path


def create_sdf(x, y, z, intensity, highest_res, mean_intensity, std_intensity, samples, filenr, savepath, data_split):
    global zdiffmax, all_samples, fileindex
    xv = x
    yv = y
    zv = z
    iv = intensity

    zmax = zv[~np.isnan(zv)].max()
    zmin = zv[~np.isnan(zv)].min()
    xmax = xv.max()
    xmin = xv.min()
    ymax = yv.max()
    ymin = yv.min()
    # print(zmax, zmin)
    
    xvflat = xv.flatten()
    yvflat = yv.flatten()
    zvflat = zv.flatten()
    
    #find center of the mesh before normalization to save in the npz file. Each sdf is indivually normalized to -1,1
    points = np.vstack((xvflat, yvflat, zvflat)).T
    nan_indices = np.where(np.isnan(points).any(axis=1))[0]
    points2 = np.delete(points, nan_indices, axis=0)
    bbmin = np.min(points2, axis=0)
    bbmax = np.max(points2, axis=0)
    center = (bbmin + bbmax) * 0.5
    if (zmax-zmin) == 0:
        return False
    scale = 2 / (zmax-zmin)
    points = (points - center) * scale

    xvflat = points[:,0]
    yvflat = points[:,1]
    zv = points[:,2].reshape(xv.shape) 
    




    x_grid_hr = np.linspace(xvflat.min(), xvflat.max(), highest_res)
    y_grid_hr = np.linspace(yvflat.min(), yvflat.max(), highest_res)
    z_grid_hr = np.linspace(-1, 1, highest_res)
    xh,yh,zh = np.meshgrid(x_grid_hr, y_grid_hr, z_grid_hr)

    hrmeterres = [(xmax-xmin)/highest_res, (ymax-ymin)/highest_res, (zmax-zmin)/highest_res]
    # print("High meter resolution: ", hrmeterres)    
    
    sdf = np.zeros_like(xh)
    # fill the HR SDF
    for i in range(xh.shape[0]):
        for j in range(xh.shape[1]):
            for k in range(xh.shape[2]):
                sdf[i,j,k] = zh[i,j,k] - zv[j,i]
    

    z_grid_lr = np.linspace(-1, 1, highest_res//2)
    xl,yl,zl = np.meshgrid(x_grid_hr[::2], y_grid_hr[::2], z_grid_lr)
    lrmeterres=[(xmax-xmin)/(highest_res/2), (ymax-ymin)/(highest_res/2), (zmax-zmin)/(highest_res/2)]


    zv_ds = zv[::2, ::2]
    # fill the LR SDF
    sdf_ds = np.zeros_like(xl)
    for i in range(xl.shape[0]):
        for j in range(xl.shape[1]):
            for k in range(xl.shape[2]):
                sdf_ds[i,j,k] = zl[i,j,k] - zv_ds[j,i]

    
    if intensity is not None:
        #maxmin normalization
        the_mean = mean_intensity 
        the_std = std_intensity  
        max_intensity = the_std*3 + the_mean
        iv[np.isnan(iv)] = 0
        iv_ds = iv[::2, ::2]
        iv_ds = iv_ds/max_intensity
        iv_layers = np.repeat(iv_ds[:, :, np.newaxis], 32, axis=2)

        
    
    
    
    
    
    if highest_res == 128:
        #if highest resolution is 128, we need to create a second downsampled version so we fill another LR SDF This one is 32x32x32 the other LR SDF is 64x64x64 and the HR SDF is 128x128x128
        z_grid_lr2 = np.linspace(-1, 1, highest_res//4)
        xl2,yl2,zl2 = np.meshgrid(x_grid_hr[::4], y_grid_hr[::4], z_grid_lr2)
        lrmeterres2=[(xvflat.max()-xvflat.min())/(highest_res/4), (yvflat.max()-yvflat.min())/(highest_res/4), (zmax-zmin)/(highest_res/4)]
       
        zv_ds2 = zv[::4, ::4]

        sdf_ds2 = np.zeros_like(xl2)
        for i in range(xl2.shape[0]):
            for j in range(xl2.shape[1]):
                for k in range(xl2.shape[2]):
                    sdf_ds2[i,j,k] = zl2[i,j,k] - zv_ds2[j,i]

        if intensity is not None:
            iv_ds = iv[::4, ::4]
            iv_ds2 = iv_ds2/max_intensity
            iv_layers2 = np.repeat(iv_ds2[:, :, np.newaxis], 32, axis=2)
    


    

    
    

    if highest_res != 128:
        # create folder if it does not exist
        if not os.path.exists(savepath + "file" + str(filenr)):
            os.makedirs(savepath + "file" + str(filenr))
        if samples < 10:
            savename = "sdf_000" + str(samples)+ ".npz"
        else:
            savename = "sdf_00" + str(samples)+ ".npz"
        savepath = Path(savepath + "file" + str(filenr) +"/" + savename)
        if intensity is not None:
            np.savez(savepath, sdf=sdf, sdf_ds=sdf_ds, hrmeterres=hrmeterres, lrmeterres=lrmeterres, center = center, intensity = iv_layers, data_split = data_split)
        else:
            np.savez(savepath, sdf=sdf, sdf_ds=sdf_ds, hrmeterres=hrmeterres, lrmeterres=lrmeterres, center = center, data_split = data_split, intensity = [])
    else:
        # create folder if it does not exist
        if not os.path.exists(savepath + "file" + str(filenr) + str(filenr)):
            os.makedirs(savepath + "file" + str(filenr))
        if samples < 10:
            savename = "sdf_000" + str(samples)+ ".npz"
        else:
            savename = "sdf_00" + str(samples)+ ".npz"
        savepath = Path(savepath + "file" + str(filenr) +"/" + savename)
        if intensity is not None:
            np.savez(savepath, sdf=sdf, sdf_ds=sdf_ds, sdf_ds2 = sdf_ds2, hrmeterres=hrmeterres, lrmeterres=lrmeterres, lrmeterres2=lrmeterres2, center = center, data_split = data_split, intensity = iv_layers2)
        else:
            np.savez(savepath, sdf=sdf, sdf_ds=sdf_ds, sdf_ds2 = sdf_ds2, hrmeterres=hrmeterres, lrmeterres=lrmeterres, lrmeterres2=lrmeterres2, center = center, data_split = data_split, intensity = [])

    

    
    return True












data = np.load("clean_data.npz")
zdiffmax = max(data["zdifflist"])

all_samples = data["all_samples"]
print(len(all_samples))
fileindex = data["fileindex"]
print(len(all_samples[sum(fileindex[0:88-24]):sum(fileindex)]))
print(len(all_samples[0:sum(fileindex[:80-24])]))

def main():
    global zdiffmax, all_samples, fileindex
    path = Path("../data/the_one")
    savepath = "../data/clean_data/"
    files = path.glob("*.npz")
    files = sorted(files)
    starting_filenr = 12
    filenr = starting_filenr
    samples = 0
    done = False
    test_data_cutoff = data["test_data_cutoff"]
    val_data_cutoff = data["val_data_cutoff"]
    only_test_gen = data["only_test_gen"]
    intensity_data = True
    try:
        mean_intensity = data["mean_intensity"]
        std_intensity = data["std_intensity"]
    except KeyError:
        print("No mean and std intensity found, dataset will not contain intensity")
        mean_intensity = None
        std_intensity = None
        intensity_data = False
    for name in files: 
        current_data = np.load(name)
        if filenr < test_data_cutoff:
            res = data["slice_size_test"]
            data_split = "test"
        elif filenr > val_data_cutoff:
            res = data["slice_size_train_and_val"]
            data_split = "val"
            if only_test_gen:
                filenr += 1
                continue
        else:
            res = data["slice_size_train_and_val"]
            data_split = "train"
            if only_test_gen:
                filenr += 1
                continue
        print("filenr: ", filenr)
        file_data = all_samples[sum(fileindex[:filenr-starting_filenr]):sum(fileindex[:filenr+1-starting_filenr])]
        # print(len(file_data))
        samples = 0
        for data_sample in file_data:
            if intensity_data:
                x,y,z,i = data_sample
            else:
                x,y,z = data_sample
                i = None
            done = create_sdf(x,y,z,i,res, mean_intensity, std_intensity, samples, filenr, savepath, data_split)
            if done:
                samples += 1
        filenr += 1

main()
