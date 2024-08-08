import numpy as np
import warnings
from pathlib import Path




def collect_data(data, nan_percentage, slice_size, similar_count_th, threshold, resolution, data_slice):
    global zdifflist, fileindex, all_samples, my_dict
    x = data['X']
    y = data['Y']
    z = data['Z']
    x_grid = np.arange(x.min(), x.max(), resolution)
    y_grid = np.arange(y.min(), y.max(), resolution)
    xv, yv = np.meshgrid(x_grid, y_grid)

    zv = np.zeros_like(xv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(xv.shape[0]):
            for j in range(xv.shape[1]):

                zv[i, j] = z[(x > (xv[i, j] - resolution/2)) & (x < (xv[i, j] + resolution/2)) & (y > (yv[i, j] - resolution/2)) & (y < (yv[i, j] + resolution/2))].mean()
    
    
    # Randomly sample a 64x64 slice from zv
    samples = 0
    slice_samples = []
    similar_count = 0
    print("zv shape: ", zv.shape)
    while True:
        if zv.shape[0] < slice_size or zv.shape[1] < slice_size:
            break
        x_slice = np.random.randint(0, zv.shape[0] - slice_size)
        y_slice = np.random.randint(0, zv.shape[1] - slice_size)
    

        z_slice = zv[x_slice:x_slice+slice_size, y_slice:y_slice+slice_size]
        #find number of nan values in the slice
        nan_count = np.isnan(z_slice).sum()
        if nan_count > nan_percentage * slice_size**2:
            continue
        similar = False
        for sample in slice_samples:
            SI = max(0,min(y_slice+slice_size,sample[1]+slice_size) - max(y_slice,sample[1])) * max(0,min(x_slice+slice_size,sample[0]+slice_size) - max(x_slice,sample[0]))
            SU = slice_size**2 + slice_size**2 - SI
            if SI/SU > threshold: 
                similar = True
                similar_count += 1
                break      


        if similar_count > similar_count_th:
            break
        
        
        if not similar:
            similar_count = 0
            samples+=1
            all_samples.append((xv[x_slice:x_slice+slice_size, y_slice:y_slice+slice_size], yv[x_slice:x_slice+slice_size, y_slice:y_slice+slice_size], z_slice.copy()))
            diff = np.nanmax(z_slice) - np.nanmin(z_slice)
            if data_slice == "test":
                my_dict["test"].append((x_slice, y_slice))
            elif data_slice == "val":
                my_dict["val"].append((x_slice, y_slice))
            elif data_slice == "train":
                my_dict["train"].append((x_slice, y_slice))
        
            zdifflist.append(diff)
            slice_samples.append((x_slice, y_slice))
  


    fileindex.append(samples)
    print(samples)
    return samples

                


        
        




all_samples = []
my_dict = {
    "test": [],
    "val": [],
    "train": []
}
zdifflist = []
fileindex = []
def main():
    global zdifflist, all_samples, fileindex
    # data = np.load("../data/the_one/EM2040-0012-pumpco2o-20220119-154234.xyz.npz")
    path = Path("../data/the_one")
    files = path.glob("*.npz")
    files = sorted(files)
    total_samples = 0
    starting_filenr = 12 # the first file number                                                                   |          
    test_data_cutoff = 25 #pick files up to this number as test data                                               |
    val_data_cutoff = 79 #pick files after this number as validation data                                          |
    nan_percentage = 0.5 #how much percentage of nans in each slice is acceptable?                                 |                                    
    similar_count_th = 60 #how many times a similar slice can be generated before moving on to the next file       |        THESE ARE THE PARAMETERS YOU CAN CHANGE
    meter_res_test = 5 #if you half res you need to double the slice size                                          |                                   
    slice_size_test = 64 #                                                                                         |
    meter_res_train_and_val = 5 #                                                                                  |
    slice_size_train_and_val = 64 #                                                                                |
    only_test_gen = False #if True, only test data will be generated                                               | 
    data_slice = "none"
    filenr = starting_filenr
    for name in files:
        if filenr < test_data_cutoff: #test_data
            th = 0.15
            meter_res = meter_res_test
            slice_size = slice_size_test
            data_slice = "test"
        elif filenr <= val_data_cutoff: #train_data
            th= 0.5
            meter_res = meter_res_train_and_val
            slice_size = slice_size_train_and_val
            data_slice = "train"
            if only_test_gen:
                filenr += 1
                continue
        else: #val_data
            th = 0.5
            meter_res = meter_res_train_and_val
            slice_size = slice_size_train_and_val
            data_slice = "val"
            if only_test_gen:
                filenr += 1
                continue

        print("File: ", name)
        print("test_length:" , len(my_dict["test"]))
        print("val_length:" , len(my_dict["val"]))
        print("train_length:" , len(my_dict["train"]))
        data = np.load(name)
        total_samples += collect_data(data, nan_percentage, slice_size, similar_count_th, th, meter_res, data_slice)
        filenr += 1
    print("Total samples: ", total_samples)
    all_samples = np.array(all_samples)
    zdifflist = np.array(zdifflist)
    fileindex = np.array(fileindex)
    np.savez("clean_data.npz", all_samples=all_samples, zdifflist=zdifflist, fileindex=fileindex, test_data_cutoff=test_data_cutoff, val_data_cutoff=val_data_cutoff, meter_res_test=meter_res_test, slice_size_test=slice_size_test, meter_res_train_and_val=meter_res_train_and_val, slice_size_train_and_val=slice_size_train_and_val, only_test_gen=only_test_gen)
        
main()
