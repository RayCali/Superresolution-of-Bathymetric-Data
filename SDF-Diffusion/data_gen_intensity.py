import numpy as np
import warnings
from pathlib import Path


def create_sdf(x, y, z, resolution, all_iv, all_zv, highest_res = 64, test=False):
    if test:
        x_grid = np.arange(x.min(), x.max(), resolution)
        x_grid = x_grid[0:highest_res]
        y_grid = np.arange(y.min(), y.max(), resolution)
        y_grid = y_grid[0:highest_res]

        xv, yv = np.meshgrid(x_grid, y_grid)

        zv = np.zeros_like(xv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i in range(xv.shape[0]):
                for j in range(xv.shape[1]):

                    zv[i, j] = z[(x > (xv[i, j] - resolution/2)) & (x < (xv[i, j] + resolution/2)) & (y > (yv[i, j] - resolution/2)) & (y < (yv[i, j] + resolution/2))].mean()
    else:
        xv = x
        yv = y
        zv = z
        iv = np.zeros_like(xv)
        rows,cols = all_zv.shape
        #find the corresponding intensity value for the generated slice in data_gen.py
        for i in range(rows- (highest_res-1)):
            for j in range(cols- (highest_res-1)):
                if np.array_equal(zv, all_zv[i:i+highest_res, j:j+highest_res], equal_nan=True):
                    iv = all_iv[i:i+highest_res, j:j+highest_res]
                    print("found")
                    break
    

    

    
    return iv


def calculate_mean_and_std_intensity(files, test_data_cutoff,val_data_cutoff):
    #calculate mean and std for traindata for outlier removal later
    intensity_mean = []
    intensity_std = []
    filenr = 12
    for name in files:
        if filenr < test_data_cutoff or filenr > val_data_cutoff:
            filenr += 1
            continue
        data = np.load(name)
        x = data['X']
        y = data['Y']
        z = data['Z']
        intensity = data['intensity']
        intensity_mean.append(intensity.mean())
        intensity_std.append(intensity.std())
        filenr += 1
        total_mean = np.mean(intensity_mean)
        total_std = np.std(intensity_std)

    return total_mean, total_std













def main():
    loaded_data = np.load("clean_data.npz")
    all_samples = loaded_data["all_samples"]
    fileindex = loaded_data["fileindex"]
    test_data_cutoff = loaded_data["test_data_cutoff"]
    val_data_cutoff = loaded_data["val_data_cutoff"]
    
    path = Path("../data/the_one")
    files = path.glob("*.npz")
    files = sorted(files)
    starting_filenr = 12
    filenr = starting_filenr
    samples = 0
    new_all_samples = []
    meter_res_test = loaded_data["meter_res_test"]
    meter_res_train_and_val = loaded_data["meter_res_train_and_val"]
    size_test = loaded_data["slice_size_test"]
    size_train_and_val = loaded_data["slice_size_train_and_val"]
    only_test_gen = data["only_test_gen"]
    zscore_std = 3
    mean_intensity, std_intensity = calculate_mean_and_std_intensity(files, test_data_cutoff,val_data_cutoff)
    
    for name in files: 
        data = np.load(name)
        if filenr < test_data_cutoff:
            res = meter_res_test
            slice_size = size_test

        else:
            if only_test_gen:
                filenr += 1
                continue
            res = meter_res_train_and_val
            slice_size = size_train_and_val
        
        intensity = data['intensity']
        x = data['X']
        y = data['Y']
        z = data['Z']
        x_grid = np.arange(x.min(), x.max(), res)
        y_grid = np.arange(y.min(), y.max(), res)



        print("removing outliers")
        the_mean = mean_intensity
        the_std = std_intensity
        zscore = (intensity -  the_mean) / the_std
        outliers = np.abs(zscore) > zscore_std     #z score outlier removal
        
        intensity[outliers] = the_std*zscore_std + the_mean
        print(intensity.shape)
        print("averaging intensity")
        


        xv, yv = np.meshgrid(x_grid, y_grid)
        iv = np.zeros_like(xv)
        zv = np.zeros_like(xv)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i in range(xv.shape[0]):
                for j in range(xv.shape[1]):
                    zv[i,j] = z[(x > (xv[i, j] - res/2)) & (x < (xv[i, j] + res/2)) & (y > (yv[i, j] - res/2)) & (y < (yv[i, j] + res/2))].mean()
                    iv[i, j] = intensity[(x > (xv[i, j] - res/2)) & (x < (xv[i, j] + res/2)) & (y > (yv[i, j] - res/2)) & (y < (yv[i, j] + res/2))].mean()
        
        print("filenr: ", filenr)
        file_data = all_samples[sum(fileindex[:filenr-starting_filenr]):sum(fileindex[:filenr+1-starting_filenr])]
        print(len(file_data))
        samples = 0
        for data in file_data:
            x,y,z = data
            new_all_samples.append((all_samples[sum(fileindex[:filenr-starting_filenr])+samples][0], all_samples[sum(fileindex[:filenr-starting_filenr])+samples][1], all_samples[sum(fileindex[:filenr-starting_filenr])+samples][2], create_sdf(x, y, z, res, iv, zv, highest_res=slice_size)))
            
            samples += 1
        filenr += 1
    np.savez("clean_data_intensity.npz", all_samples=new_all_samples, only_test_gen= only_test_gen, fileindex=fileindex, zdifflist=loaded_data["zdifflist"],test_data_cutoff=test_data_cutoff, val_data_cutoff=val_data_cutoff, meter_res_test=meter_res_test, slice_size_test=loaded_data["slice_size_test"], meter_res_train_and_val=meter_res_train_and_val, slice_size_train_and_val=loaded_data["slice_size_train_and_val"], mean_intensity=mean_intensity, std_intensity=std_intensity)

main()
