import numpy as np
import os, glob, cf, math, csv, tqdm, random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime as dt
import subprocess as sp
from multiprocessing import Process
import multiprocessing as mp
from tensorflow.keras import models, layers, regularizers, callbacks, initializers, metrics
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot

def parse_data(ibtracs_path: str = "/home/users/dgalea/ibtracs.ALL.list.v04r00.csv", start_date: dt = dt(1979, 1, 1, 0), end_date: dt = dt(2017, 7, 31, 23, 59)) -> dict:

    '''
    Parse IBTrACS CSV to get a dict of TCs
    args:
        ibtracs_path: path to IBTrACS csv
        start_date: date to start processing from
        end_date: date to end processing at
    return:
        tc_dict: dict holding TC information at every timestep
    '''
    
    # set up dict
    tc_dict = {}

    # create keys
    curr_date = start_date
    while curr_date <= end_date:
        tc_dict[curr_date] = []
        curr_date += datetime.timedelta(hours=6)		

    # open IBTrACS
    with open(ibtracs_path) as csvfile:

        # read IBTrACS data
        readCSV = csv.reader(csvfile, delimiter=',')

        # loop over each row
        for i, row in enumerate(readCSV):

            # skip over header
            if i < 2:
                continue

            # process date
            date, time = row[6].split(" ")
            year, month, day = map(int, date.split("-"))
            hour = int(time.split(":")[0])
            date_time = datetime.datetime(year, month, day, hour)

            # only keep data every 6 hours
            if date_time < start_date or date_time > end_date or hour%6 != 0:
                continue

            # get TC attributes
            storm_type = row[7]
            lat = row[8]
            lon = row[9]
            wmo_wind = row[10]
            wmo_press = row[11]
            usa_wind = row[23]
            usa_press = row[24]
            tokyo_wind = row[45]
            tokyo_press = row[46]
            cma_wind = row[57]
            cma_press = row[58]
            hko_wind = row[62]
            hko_press = row[63]
            newdelhi_wind = row[67]
            newdelhi_press = row[68]
            reunion_wind = row[75]
            reunion_press = row[76]
            bom_wind = row[95]
            bom_press = row[96]
            nadi_wind = row[120]
            nadi_press = row[121]
            wellington_wind = row[124]
            wellington_press = row[125]
            ds824_wind = row[129]
            ds824_press = row[130]
            td9636_wind = row[134]
            td9636_press = row[135]
            neumann_wind = row[144]
            neumann_press = row[145]
            mlc_wind = row[149]
            mlc_press = row[150]
            usa_cat = row[25] 

            # obtain wind value
            wind = -1
            wind_choice = [wmo_wind, usa_wind, tokyo_wind, cma_wind, hko_wind, newdelhi_wind, reunion_wind, bom_wind, nadi_wind, wellington_wind, ds824_wind, td9636_wind, neumann_wind, mlc_wind]
            for choice in wind_choice:
                if choice.strip() and wind == -1 :
                    wind = int(choice)
                    break

            # obtain pressure value
            press = -1
            press_choice = [wmo_press, usa_press, tokyo_press, cma_press, hko_press, newdelhi_press, reunion_press, bom_press, nadi_press, wellington_press, ds824_press, td9636_press, neumann_press, mlc_press]
            for choice in press_choice:
                if choice.strip() and press == -1:
                    press =	int(choice)
                    break

            # get TC category
            try:
                usa_cat = int(usa_cat)
            except:
                if storm_type[0] == "E": #post-tropical
                    usa_cat = -4
                elif storm_type in ["WV", "LO", "DB", "DS", "IN", "MD"]: #misc disturbances
                    usa_cat = -3
                elif storm_type[0] == "SS": #subtropical
                    usa_cat = -2
                if wind < 34 and storm_type[0] == "T": # tropical depression
                    usa_cat = -1
                elif 34 <= wind < 64 and storm_type[0] == "T": # tropical storm
                    usa_cat = 0
                elif 64 <= wind < 83: #cat 1
                    usa_cat = 1
                elif 83 <= wind < 96: #cat 2
                    usa_cat = 2
                elif 96 <= wind < 113: #cat 3
                    usa_cat = 3
                elif 113 <= wind < 137: #cat 4
                    usa_cat = 4
                elif wind >= 137: #cat 5
                    usa_cat = 5 
                else:
                    usa_cat = -5
                
            # if both wind speed and pressure values are available, add TC centre to dict
            if wind != -1 or press != -1:
                tc_dict[date_time].append([int(usa_cat), float(lat), float(lon), int(wind), int(press)])

    return tc_dict

def split_into_basins(tc_dict: dict, start_date: dict, end_date: dict) -> dict:

    '''
    Split a timestep into regions with labels from IBTrACS
    args:
        tc_dict: dict of TCs lats/lons etc. from IBTrACS
        start_date: date to start processing from
        end_date: date to end processing at
    returns:
        cases: dict of TC cases in region with keys being timesteps
    '''
    
    #Array boundaries and corresponding lats/lons
    limits = []
    limits.append([[42, 128, 284, 398], [-60, 0, 20, 100]]) #SI
    limits.append([[42, 128, 398, 512], [-60, 0, 100, 180]]) #SWP
    limits.append([[42, 128, 0, 114], [-60, 0, 180, 260]]) #SEP
    limits.append([[42, 128, 114, 228], [-60, 0, 260, 340]]) #SA
    limits.append([[128, 214, 284, 398], [0, 60, 20, 100]]) #NI
    limits.append([[128, 214, 398, 512], [0, 60, 100, 180]]) #WP
    limits.append([[128, 214, 0, 114], [0, 60, 180, 260]]) #EP
    limits.append([[128, 214, 114, 228], [0, 60, 260, 340]]) #WA

    #Prepare region dict
    cases = {}
    curr_date = start_date
    while curr_date <= end_date:
        cases[curr_date] = {}
        curr_date += datetime.timedelta(hours=6)

    #For each date in IBTrACS
    for i, key in enumerate(tc_dict.keys()):
        
        #Get TCs in timestep
        tcs = tc_dict[key]
        
        #For each region, check if a TC is present and add region to region dict
        for limit in limits:

            # setting up variables to get max wind speed and min MSLP
            max_cat = -1e6
            min_press = 0
            max_wind = 0

            # get boundaries of region
            limit_start_lat = limit[1][0]
            limit_end_lat = limit[1][1]
            limit_start_lon = limit[1][2]
            limit_end_lon = limit[1][3]

            # loop TCs
            for tc in tcs:

                # TC details
                cat, lat, lon, wind, press = tc
                
                # calculate max wind speed and min MSLP
                if limit_start_lat < lat < limit_end_lat and limit_start_lon < (lon + 360) % 360 < limit_end_lon:
                    if cat > max_cat:
                        max_cat = cat
                        min_press = press
                        max_wind = wind

            # add details of strongest TC in region/timestep to dict
            if max_cat == -1e6:
                cases[key][tuple([limit[1][0], limit[1][1], limit[1][2], limit[1][3]])] = ["no", "no", "no"]
            else:
                cases[key][tuple([limit[1][0], limit[1][1], limit[1][2], limit[1][3]])] = [max_cat, max_wind, min_press]

    return cases

def filter_field(field: np.array, low: np.array, high: np.array) -> np.array:

    '''
    Perform spherical filtering
    args:
        field field (as a np array) to perofrm spherical filtering on
        low: lowest wavenumber to keep (>=1)
        high: highest wavenumber to keep
    returns:
        filtered: filtered data
    '''

    # get array shape
    nlats_reg = field.shape[0]
    nlons_reg = field.shape[1]

    # calculate spherical coefficients
    reggrid = Spharmt(nlons_reg,nlats_reg,gridtype='regular')

    # regrid to spectral grid, keeping up to high wavenumber
    field_high = reggrid.grdtospec(field, high)

    # regrid high wavenumber spectral grid to lat/lon grid
    field_high = reggrid.spectogrd(field_high)

    # regrid to spectral grid, keeping up to low wavenumber
    field_low = reggrid.grdtospec(field, low)
    
    # regrid low wavenumber spectral grid to lat/lon grid
    field_low = reggrid.spectogrd(field_low)

    # filter data
    filtered = field_high - field_low

    return filtered

def field_select_area(field: np.array, limits: list[float, float, float, float]) -> np.array:

    '''
    Select a region from global data
    args:
        field: field (as a np array) to choose data from
        limits: bounds for region to choose
    returns:
        region_field: field of data for a particular region
    '''

    return field[limits[0] : limits[1], limits[2] : limits[3]]

def get_data(date: dt, limits: list[float, float, float, float], cat: int, wind_tc: float, press_tc: float, save_path: str, save: bool = True, filtered: bool = False) -> np.array:

    '''
    Function to get and save data
    args:
        date: date of region to process
        limits: bounds of region to process
        cat: max cat of TC in region
        wind_tc: max wind of TC in region
        press_tc: min MSLP of TC in region
        save_path: path to save region to
        save: option to save or not
        filtered: option to perofrm spherical filtering
    return:
        arr: numpy array of data from nc file
    '''
    
    #Load path for data
    load_path = "/badc/ecmwf-era-interim/data/gg/as/" + str(date.year) + "/" + str(date.month).zfill(2) + "/" + str(date.day).zfill(2) + "/ggas" + str(date.year).zfill(2) + str(date.month).zfill(2) + str(date.day).zfill(2) + str(date.hour).zfill(2) + "00.nc"

    # get region boundaries
    start_lat, end_lat, start_lon, end_lon = limits
    arr_limits = get_array_boundaries(limits)		

    # set up array for data
    arr = np.zeros((86, 114, 5))
    
    # read single layer nc file
    nc = cf.read(load_path)

    # get MSLP data
    press = np.roll(np.flipud(nc.select("air_pressure_at_sea_level")[0].data.array[0, 0, :, :]), (0, 256))
    if filtered:
        press = filter_field(press, 5, 106)
    press = field_select_area(press, arr_limits)

    # get wind speed
    u = nc.select("northward_wind").select_by_units("m s**-1")[0].data.array
    v = nc.select("eastward_wind").select_by_units("m s**-1")[0].data.array
    wind = np.roll(np.flipud(np.sqrt(u[0, 0, :, :]**2 + v[0,0,:,:]**2)), (0, 256))
    if filtered:
        wind = filter_field(wind, 5, 106)
    wind = field_select_area(wind, arr_limits)

    # add MSLP to data array
    arr[:,:,0] = press

    # add wind speed to data array
    arr[:,:,1] = wind
    
    # close nc file
    nc.close()

    # load multi-layer nc file
    nc = cf.read(load_path.replace("as", "ap"))

    # pressure levels at which to get vorticity
    levs = [850, 700, 600]

    # loop over pressure levels
    for i, lev in enumerate(levs):

        # get relative vorticity
        rel_vort = np.roll(np.flipud(nc.select("atmosphere_relative_vorticity").select_by_units("s**-1")[0].subspace(Z=lev).data.array[0, 0, :, :]), (0, 256))

        # filter field if needed
        if filtered:
            rel_vort = filter_field(rel_vort, 1, 63)
        if start_lat < 0:
            rel_vort = -rel_vort
        rel_vort = field_select_area(rel_vort, arr_limits)

        # add relative vorticity to data array
        arr[:,:,i+2] = rel_vort
        
    # recast array
    arr = np.float32(arr)

    # close multi-layer nc file
    nc.close()

    # save file if needed
    if save == True:
        if cat == "no":
            np.savez(save_path + "no_" + str(limits[0]) + "_" + str(limits[1]) + "_" + str(limits[2]) + "_" + str(limits[3]) + "_" + str(date.year).zfill(2) + str(date.month).zfill(2) + str(date.day).zfill(2) + str(date.hour).zfill(2) + ".npz", arr)
        else:
            np.savez(save_path + str(cat) + "_" + str(press_tc) + "_" + str(wind_tc) + "_" + str(limits[0]) + "_" + str(limits[1]) + "_" + str(limits[2]) + "_" + str(limits[3]) + "_" + str(date.year).zfill(2) + str(date.month).zfill(2) + str(date.day).zfill(2) + str(date.hour).zfill(2) + ".npz", arr)
    else:
        return arr

def get_array_boundaries(limits: list[float, float, float, float]) -> list[float, float, float, float]:

    '''
    Function to go from array indexes to lat/lon for region bounds
    args:
        limits: list of regions index bounds in [min_row, max_row, min_col, max_col]
    '''
    
    # set dict to translate row/col numer to lat/lon degree
    limits_dict = {}
    limits_dict[42, 128, 284, 398] = [-60, 0, 20, 100] #SI
    limits_dict[42, 128, 398, 512] = [-60, 0, 100, 180] #SWP
    limits_dict[42, 128, 0, 114] = [-60, 0, 180, 260] #SEP
    limits_dict[42, 128, 114, 228] = [-60, 0, 260, 340] #SA
    limits_dict[128, 214, 284, 398] = [0, 60, 20, 100] #NI
    limits_dict[128, 214, 398, 512] = [0, 60, 100, 180] #WP
    limits_dict[128, 214, 0, 114] = [0, 60, 180, 260] #EP
    limits_dict[128, 214, 114, 228] = [0, 60, 260, 340] #WA

    # return lat/lon degree for region bounds
    for key in limits_dict.keys():
        if limits[0] == limits_dict[key][0] and limits[1] == limits_dict[key][1] and limits[2] == limits_dict[key][2] and limits[3] == limits_dict[key][3]:
            return key

def create_data():

    '''
    Function to create all data
    '''
    
    # Path to IBTrACS file
    path = "/home/users/dgalea/ibtracs.ALL.list.v04r00.csv"
    
    # Path to save data to
    save_path = "/work/scratch-nopw/dg/new_ibtracs/"

    # Date bounds
    start_date = dt(1979, 1, 1)
    end_date = dt(2019, 9, 1)

    # Parse IBTrACS data
    tc_dict = parse_data(ibtracs_path=path, start_date=start_date, end_date=end_date)
    
    # Get cases
    cases = split_into_basins(tc_dict, start_date, end_date)
    
    # Prepare inputs for multiprocessing
    args = []
    for date in cases:
        for limits in cases[date]:
            cat, wind, press = cases[date][limits]
            args.append([date, limits, cat, wind, press, save_path, True, True])
    
    # Start pool of workers
    pool = mp.Pool(mp.cpu_count())
    
    # Get and save data
    pool.starmap(get_data, args, chunksize=1)
    
    # Close pool of workers
    pool.close()
        
def create_data_example():
    
    '''
    Create an example of the data (Fig1)
    '''
    
    # Load pressure level files
    ap_nc = cf.read("ggap200508281800.nc")
    
    # Load surface level file
    as_nc = cf.read("ggas200508281800.nc")

    # Select variables
    press = as_nc.select("air_pressure_at_sea_level")[0].subspace(X=cf.wi(260, 340), Y=cf.wi(0, 60)).data.array
    u = as_nc.select("northward_wind").select_by_units("m s**-1")[0].subspace(X=cf.wi(260, 340), Y=cf.wi(0, 60)).data.array
    v = as_nc.select("northward_wind").select_by_units("m s**-1")[0].subspace(X=cf.wi(260, 340), Y=cf.wi(0, 60)).data.array
    rel_vort_850 = ap_nc.select("atmosphere_relative_vorticity").select_by_units("s**-1")[0].subspace(Z=850, X=cf.wi(260, 340), Y=cf.wi(0, 60)).data.array
    rel_vort_700 = ap_nc.select("atmosphere_relative_vorticity").select_by_units("s**-1")[0].subspace(Z=700, X=cf.wi(260, 340), Y=cf.wi(0, 60)).data.array
    rel_vort_600 = ap_nc.select("atmosphere_relative_vorticity").select_by_units("s**-1")[0].subspace(Z=600, X=cf.wi(260, 340), Y=cf.wi(0, 60)).data.array

    # Get wind speed
    wind = np.sqrt(u**2 + v**2)
    
    # Place data in a numpy array
    nc_arr = np.zeros((86, 114, 5))
    nc_arr[:, :, 0] = press[0, 0, :, :]
    nc_arr[:, :, 1] = wind[0, 0, :, :]
    nc_arr[:, :, 2] = rel_vort_850[0, 0, :, :]
    nc_arr[:, :, 3] = rel_vort_700[0, 0, :, :]
    nc_arr[:, :, 4] = rel_vort_600[0, 0, :, :]
    
    # Flip to get right orientation
    nc_arr = np.flipud(nc_arr)
    
    # Load (mostly) preprocessed data
    arr = np.load("5_902_150_0_60_260_340_2005082818.npz")['arr_0']

    # Perform standardisation
    for field in range(arr.shape[-1]):
        mean_ = np.mean(arr[:,:,field])
        std_ = np.std(arr[:,:,field])
        arr[:,:,field] = (arr[:,:,field] - mean_) / std_
        
    # Plot
    fig, axs = plt.subplots(5, 2, figsize=(12, 15))
    for i in range(5):
        im = axs.flatten()[i*2].contourf(nc_arr[:, :, i])
        divider = make_axes_locatable(axs.flatten()[i*2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        axs.flatten()[i*2].set_xticks([])
        axs.flatten()[i*2].set_yticks([])
    for i in range(5):
        im = axs.flatten()[i*2+1].contourf(arr[:, :, i])
        divider = make_axes_locatable(axs.flatten()[i*2+1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        axs.flatten()[i*2+1].set_xticks([])
        axs.flatten()[i*2+1].set_yticks([])

    # Set titles
    axs[0, 0].set_title("Original Data")
    axs[0, 1].set_title("Preprocessed Data")

    # Set ylabels
    axs[0, 0].set_ylabel("Mean Sea Level Pressure", rotation=90, labelpad=2)
    axs[1, 0].set_ylabel("Wind Speed", rotation=90, labelpad=2)
    axs[2, 0].set_ylabel("Vorticity at 850hPa", rotation=90, labelpad=2)
    axs[3, 0].set_ylabel("Vorticity at 700hPa", rotation=90, labelpad=2)
    axs[4, 0].set_ylabel("Vorticity at 600hPa", rotation=90, labelpad=2)

    # Save figure
    plt.tight_layout()
    plt.savefig("data_example.pdf")
    
def dataset_split(path: str, years: list[int]):

    '''
    Get number of positive/negative cases in the years wanted from the files available
    args:
        path: path where data is saved to
        years: list of years to consider
    '''
    
    # get list of files
    all_files = sorted(glob.glob(os.path.join(path, "*")))
    
    # counters
    no = 0
    yes = 0
    
    # for each file, add to respective counter
    for file in all_files:
        name = file.split("/")[-1].replace(".npz", "")
        date = name.split("_")[-1]
        year = int(date[:4])
        
        if year in years:
            cat = name.split("_")[0]
            if cat == "no":
                no += 1
            else:
                cat = int(cat)
                if cat < 1:
                    no += 1
                else:
                    yes += 1
    
    return yes, no

def train_dl_model(train_data: np.array, train_labels: np.array, val_data: np.array, val_labels: np.array, model_name: str) -> tf.keras.Model:

    '''
    Train the DL model using the data and labels given
    args:
        train_data: data for training
        train_labels: labels for data in train_data
        val_data: data for validation
        val_labels: labels for data in val_data
        model_name: save name for model
    returns:
        model: trained model
    '''
    
    class DataGenerator(tf.keras.utils.Sequence):
    
        'Generates data for Keras'
        def __init__(self, data: np.array, labels: np.array, batch_size: int = 32, shuffle: bool = True) -> None:

            # store parameters
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.data = data
            self.labels = labels

            # split labels by TC/no TC
            negative_indices = []
            positive_indices = []
            for i in range(len(self.labels)):
                if self.labels[i] == 0:
                    negative_indices.append(i)
                else:
                    positive_indices.append(i)

            # store labels
            self.negative_indices = negative_indices
            self.positive_indices = positive_indices

            self.on_epoch_end()

        def __len__(self) -> int:
            '''
            Denotes the number of batches per epoch
            args:
                None
            returns:
                len: number of batches
            '''

            return int(np.floor(len(self.positive_indices)*2 / self.batch_size))

        def __getitem__(self, index: int) -> tuple(np.array, np.array):

            '''
            Generate one batch of data
            args:
                index: index of batch to get
            returns:
                data: training data
                labels: label data
            '''

            # Generate indexes of the batch
            indexes = self.positive_indices[int(index*self.batch_size/2):int((index+1)*self.batch_size/2)]
            indexes += self.negative_indices[int(index*self.batch_size/2):int((index+1)*self.batch_size/2)]

            # Generate data
            X, y = self.__data_generation(indexes)

            return X, y

        def on_epoch_end(self) -> None:

            'Updates indexes after each epoch'

            if self.shuffle == True:
                np.random.shuffle(self.negative_indices)
                np.random.shuffle(self.positive_indices)

        def __data_generation(self, indices: int) -> tuple(np.array, np.array):

            '''
            Generates data containing batch_size samples
            args:
                indices: indices of barches to process
            returns:
                data: training data
                labels: label data
            '''

            # data shape
            shape = self.data.shape

            # set up arrays for data and labels
            X = np.zeros((self.batch_size, shape[1], shape[2], shape[3]), dtype="float32")
            y = np.zeros(self.batch_size)

            # get data and labels
            for i in range(len(indices)):
                X[i] = self.data[indices[i]]
                y[i] = self.labels[indices[i]]

            # data augmentation
            for i in range(X.shape[0]):

                if random.random() < data_aug_rate:

                    choice = np.random.random_integers(low=1, high=3)

                    if choice == 1:
                        X[i] = np.roll(X[i], int(random.random()*X[i].shape[1]), axis=1)
                    elif choice == 2:
                        angle = random.uniform(0, 360)
                        X[i] = scipy.ndimage.rotate(X[i], angle, mode="mirror", reshape=False)
                    elif choice == 3:
                        X[i] = np.fliplr(X[i])

            return X, y
        
    def create_model(shape: list[int, int, int]) -> tf.keras.Model:

        '''
        Create DL model
        args:
            shape: shpe of training data
        returns:
            model: DL model
        '''

        # model structure
        model = models.Sequential( 
            [
                layers.Conv2D(8, (2, 2), strides=(1, 1), input_shape=(shape)),
                layers.ReLU(),
                layers.Dropout(dropout_rate),
                layers.MaxPooling2D((2, 2), strides=1), 
                layers.Conv2D(16, (2, 2), strides=(1, 1)), 
                layers.ReLU(),
                layers.Dropout(dropout_rate),
                layers.MaxPooling2D((2, 2), strides=1),
                layers.Conv2D(32, (2, 2), strides=(1, 1)), 
                layers.ReLU(),
                layers.Dropout(dropout_rate),
                layers.MaxPooling2D((2, 2), strides=1),
                layers.Conv2D(64, (2, 2), strides=(1, 1)), 
                layers.ReLU(),
                layers.Dropout(dropout_rate),
                layers.MaxPooling2D((2, 2), strides=1),
                layers.Conv2D(128, (2, 2), strides=(1, 1)), 
                layers.ReLU(),
                layers.Dropout(dropout_rate),
                layers.MaxPooling2D((2, 2), strides=1),

                layers.Flatten(),
                layers.Dense(128, kernel_regularizer=regularizers.l2(l2_val)),
                layers.ReLU(),
                layers.Dropout(dropout_rate),
                layers.Dense(64, kernel_regularizer=regularizers.l2(l2_val)),
                layers.ReLU(),
                layers.Dropout(dropout_rate),
                layers.Dense(32, kernel_regularizer=regularizers.l2(l2_val)),
                layers.ReLU(),
                layers.Dropout(dropout_rate),
                layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_val)) 
            ]
        )

        # metrics
        METRICS = [
              metrics.TruePositives(name='tp'),
              metrics.FalsePositives(name='fp'),
              metrics.TrueNegatives(name='tn'),
              metrics.FalseNegatives(name='fn'), 
              metrics.BinaryAccuracy(name='accuracy'),
              metrics.Precision(name='precision'),
              metrics.Recall(name='recall'),
              metrics.AUC(name='auc', curve='PR'),
        ]

        # optimizer
        opt = tf.keras.optimizers.SGD(momentum = mom, learning_rate = lr)

        # compile model
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=METRICS) 

        return model
    
    # hyperparameters
    epochs = 150
    batch_size = 8
    early_stop_patience = 10
    lr = 0.01
    mom = 0.8
    data_aug_rate = 0.6
    dropout_rate = 0.1
    l2_val = 0.005
    
    # Set up generator inputs
    params_train = {'data': train_data, 'labels': train_labels, 'batch_size': batch_size, 'shuffle': True}
    params_val = {'data': val_data, 'labels': val_labels, 'batch_size': batch_size, 'shuffle': True}

    # Generators
    train_gen = DataGenerator(**params_train)
    val_gen = DataGenerator(**params_val)

    # Initialise new model
    model = create_model(train_data.shape[1:])

    # Early Stopping callback
    early_stop = callbacks.EarlyStopping(monitor="val_auc", patience=early_stop_patience, mode="max", restore_best_weights=True)

    # Train model on dataset
    history = model.fit_generator(train_gen, validation_data=val_gen, shuffle=True, use_multiprocessing=False, workers=int(0.5*mp.cpu_count()),
                                      callbacks=[early_stop], verbose=2, epochs=epochs, max_queue_size = int(len(train_data)/batch_size))

    # Save model
    model.save(fold_model_name)
    
    return model

def load_dl_data(path: str, years: list[int] = None, whole_world: bool = True):

    '''
    Function to load and return all data needed for DL testing, i.e. between 1st August 2017 and 31st August 2019
    
    args:
        path: path to save location
    returns:
        data: cases from opened files as one numpy array; in the same order as files
        labels: label for each case in data
    '''
    
    #Get list of all files
    all_files=list(sorted(glob.glob(os.path.join(path, "*.npz"))))
    
    # only get list of WAWP files if needed
    if whole_world == False:
        files = []
        for file in all_files:
            if "0_60_260_340" in file or "0_60_100_180" in file:
                files.append(file)
        all_files = files

    # only add files in required dates  
    if years != None:
        files = []
        for file in all_files:
            name = file.split("/")[-1].replace(".npz", "")
            date = name.split("_")[-1]
            year = int(date[:4])
            if year in years:
                files.append(file)
        all_files = files
    
    # Open pool for multiprocessing
    pool = mp.Pool(int(0.5*mp.cpu_count()))
    
    # Get cases
    results = list(tqdm.tqdm(pool.imap(get_dl_file, all_files), total=len(all_files)))
    
    # Close pool after all cases loaded
    pool.close()
    
    # Set up numpy array to hold all cases
    shape = results[0][0].shape
    data = np.zeros((len(results), shape[0], shape[1], shape[2]))
    labels = np.zeros(len(results))
    
    # Loop through all cases and place in numpy array holding all cases
    for i, res in enumerate(results):
        data[i], labels[i] = res
    
    # Return the list of files opened and their case data
    return data, labels, all_files

def rebin(a: np.array, shape: list[int], dim: int) -> np.array:

    '''
    Function to resize (reduce) an array to a given shape
    args:
        a: 3D array to be shaped
        shape: 2D shape of new array
        dim: number of channels [== a.shape[-1]]
    returns:
        new_array: a in the shape needed
    '''

    sh = shape[0],dim,shape[1],dim
    return a.reshape(sh).mean(-1).mean(1)


def reduce_dim(dim: int, big_arr: np.array) -> np.array:

    '''
    Function to reduce an array to a given 1D factor
    args:
        dim: factor by which the array is to be resized
        big_arr: array to be reduced in size
    returns
        small_arr: reduced array
    '''
    
    # if dim == 1, then no need to resize
    if dim == 1:
        return big_arr
    
    # pad array to get right shape for resizing
    if dim == 3:
        big_arr = np.pad(big_arr, ((0, 4), (0, 0), (0, 0))) 
    elif dim == 4:
        big_arr = np.pad(big_arr, ((0, 2), (0, 2), (0, 0))) 
    elif dim == 5:
        big_arr = np.pad(big_arr, ((0, 4), (0, 1), (0, 0))) 
    
    # compute shape of resized array
    small_shape = (big_arr.shape[0]//dim, big_arr.shape[1]//dim, big_arr.shape[2])

    # create resized array, initilized as zeros
    small_arr = np.zeros(small_shape)

    # resize each channel and place in array
    for i in range(big_arr.shape[-1]):
        small_arr[:,:,i] = rebin(big_arr[:,:,i], small_shape[0:-1], dim)

    return small_arr

def get_dl_file(file: str) -> tuple(np.array, int):

    '''
    Function to load a single case for DL testing
    args:
        file: path of file to load
    returns:
        arr: case that file holds, in a numpy array of shape (22, 29, 5)
    '''    
    
    # Get file
    arr = np.load(file)['arr_0']
    
    # Reduce to a sixteenth (4*4) of ERAI resolution
    arr = reduce_dim(4, arr)
        
    # Standardisation
    for field in range(arr.shape[-1]):
        mean_ = np.mean(arr[:,:,field])
        std_ = np.std(arr[:,:,field])
        arr[:,:,field] = (arr[:,:,field] - mean_) / std_
    
    # get label based on file name
    name = file.split("/")[-1].replace(".npz", "")
    cat = name.split("_")[0]
    if cat == "no":
        label = 0
    else:
        cat = int(cat)
        if cat < 1:
            label = 0
        else:
            label = 1
    
    return arr, label

def eval_model(model: tf.keras.Model, data: np.array, labels: np.array) -> tuple(float, float, float, float, float, float, float, float, float, float):

    '''
    Evaluate DL model
    args:
        model: DL model
        data: numpy array of all data to use for evaluation
        labels: labels for all cases in data
    '''

    # get model inferences
    results = model.evaluate(data, labels, use_multiprocessing=True, workers=mp.cpu_count(), verbose=0, return_dict=True)
    
    # get metrics
    loss = results['loss']
    acc = results['accuracy']
    tp = results['tp']
    fp = results['fp']
    fn = results['fn']
    tn = results['tn']
    recall = results['recall']
    precision = results['precision']
    auc = results['auc']
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    return loss, acc, tp, fp, fn, tn, recall, precision, auc, f1

def get_conf_matrix(data: np.array, labels: np.array, model: tf.keras.Model, print_matrix: bool = True) -> None:
    
    '''
    Print confusion matrix given a DL model, data and labels
    args:
        data: data to use
        labels: labels corresponding to each case in data
        model: DL model to evaluate
        print_matrix: option to print confusion matrix
    '''
    
    # evalute model
    loss, acc, tp, fp, fn, tn, recall, precision, auc, f1 = eval_model(model, data, labels)
    
    #get number of samples
    num_samples = len(labels)
    
    if print_matrix:
        
        #Print confusion matrix
        print("Confusion Matrix:")
        print("True label, true prediction: ", tp, " of ", num_samples, ": ", "{:.2f}%".format(tp/num_samples*100))
        print("True label, false prediction: ", fn, " of ", num_samples, ": ", "{:.2f}%".format(fn/num_samples*100))
        print("False label, true prediction: ", fp, " of ", num_samples, ": ", "{:.2f}%".format(fp/num_samples*100))
        print("False label, false prediction: ", tn, " of ", num_samples, ": ", "{:.2f}%".format(tn/num_samples*100))
        print("")
        
def get_aucpr_curve(model: tf.keras.Model, test_data: np.array, test_labels: np.array) -> None:
    
    '''
    Plot AUC-PR curve for the DL model and data supplied
    args:
        model: DL model to evaluate
        data: nmupy array of data to use
        labels: labels for all cases in data
    '''

    # get model inferences
    pred_labels = model.predict(test_data)

    # set up thresholds array
    thresholds = np.zeros(11)

    # calculate and store thresholds to use
    for i in range(len(thresholds)):
        thresholds[i] = i/(len(thresholds)-1)

    # set up results arrays for precision and recall
    precision = np.zeros(len(thresholds))
    recall = np.zeros(len(thresholds))

    # loop over each threshold
    for thresh_i, thresh in enumerate(thresholds):

        # set up counters
        label_false_pred_false = 0
        label_false_pred_true = 0
        label_true_pred_true = 0
        label_true_pred_false = 0

        # loop over each inference and add to right counter
        for i in range(len(test_labels)):
            if test_labels[i] == 0 and pred_labels[i] < thresh:
                label_false_pred_false += 1
            elif test_labels[i] == 0 and pred_labels[i] >= thresh:
                label_false_pred_true += 1
            elif test_labels[i] == 1 and pred_labels[i] < thresh:
                label_true_pred_false += 1
            elif test_labels[i] == 1 and pred_labels[i] >= thresh:
                label_true_pred_true += 1

        # calculate precision
        if label_true_pred_true + label_false_pred_true == 0:
            precision[thresh_i] = 100
        else:
            precision[thresh_i] = label_true_pred_true / (label_true_pred_true + label_false_pred_true) * 100.0

        # calculate recall
        if label_true_pred_true + label_true_pred_false == 0:
            recall[thresh_i] = 0.0
        else:
            recall[thresh_i] = label_true_pred_true / (label_true_pred_true + label_true_pred_false) * 100.0

    # plot AUC-PR curve
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(recall, precision)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.set_aspect(aspect="equal")
    for i, txt in enumerate(thresholds):
        ax.annotate(txt, (recall[i], precision[i]))
    
    plt.tight_layout()
    plt.savefig("whole_world_result.pdf")
    
def imagenet_comp(path: str) -> None:

    '''
    Plot standard model comparison
    args:
        path: save path for figure
    '''
    
    # list of indices
    ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # plot
    fig, ax1 = plt.subplots(1,figsize=(7.5, 5))

    # auc values
    auc = (0.7173, 0.5489, 0.5447, 0.5430, 0.5397, 0.5369, 0.5307, 0.5194, 0.5193, 0.5187, 0.5139, 0.4955, 0.4949, 0.4940, 0.4874, 0.4765, 0.4461)

    # width of plot lines
    width = 0.75

    # plot auc
    ax1.bar(ind, auc, width, color='r')

    # formatting
    ax1.set_ylabel('AUC-PR', color='r')  
    ax1.set_xticks(ind)
    ax1.set_xticklabels(['Original', 'DenseNet121', 'ResNet50V2', 'ResNet152', 'ResNet101', 
                        'VGG16', 'DenseNet169', 'InceptionV3', 'Xception', 'VGG19',
                         'MobileNet', 'ResNet101V2', 'ResNet50', 'DenseNet201', 'InceptionResNetV2',
                         'ResNet152V2', 'MobileNetV2'], rotation='vertical')
    ax1.set_ylim([0.4, .75])

    # loss values
    loss = [0.2650, 0.4865, 0.4766, 0.5364, 0.5560, 0.9542, 0.6612, 0.5651, 0.4555, 0.9527, 0.6264, 0.5381, 1.0428, 0.6248, 0.5095, 0.4928, 0.5182]

    # plot
    axes2 = ax1.twinx()
    axes2.plot(ind, loss, 'bo')
    axes2.set_ylabel('Loss', color='b')
    axes2.set_ylim([0.2, 1.1])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(path, dpi=300)

def brieman_feature_importance(data: np.array, fields: list) -> np.array:

    '''
    Permute data across all examples in field/s to be able to perform Briemann feature importance
    args:
        data: original data
        fields: list of indices for fields to permute. 
    '''

    # set up array for permuted data
    permute_data = np.zeros_like(data)
    
    # loop each field in data
    for field_i in range(data.shape[-1]):

        # shuffle case indices
        indices = list(range(0, len(data)))
        if field_i in fields:
            random.shuffle(indices)
            
        # shuffle data
        for i in range(len(data)):
            permute_data[i, :, :, field_i] = data[indices[i], :, :, field_i]
            
    return permute_data

def lak_feature_importance(model: tf.keras.Model, data: np.array, labels: np.array, fields: list, iterations: int) -> list:
    '''
    Perform Lakshmanan feature importance
    args:
        model: DL model
        data: original data
        labels: labels for data
        fields: list of field names
        iterations: number of iteration to then take an average of
    '''
    
    # set up list for results
    ordered_fields = []
    
    # get number of fields to test
    fields_to_test = list(range(data.shape[-1]))

    # set up list for results
    finished_fields = []
    
    # loop for each pass
    for pass_i in range(data.shape[-1]):
        
        print("Pass ", pass_i+1)
        pass_auc = []
        
        # loop over each field to test
        for fields_i in range(len(fields_to_test)):
            
            # collect fields
            field_to_test = []
            field_to_test.append(fields_to_test[fields_i])
            for finished_field in finished_fields:
                field_to_test.append(finished_field)
              
            # get metrics for specified number of random bootstrapped iterations
            iter_auc = []
            for iter_i in range(iterations):
                loss, acc, tp, fp, fn, tn, recall, precision, auc, f1 = eval_model(model, brieman_feature_importance(data, field_to_test), labels)
                iter_auc.append(auc)

            # store mean AUC-PR
            pass_auc.append(np.mean(np.array(iter_auc)))
            
        # find most important field
        imp_field = np.argmin(np.array(pass_auc))
        ordered_fields.append([fields[imp_field], pass_auc[imp_field]])
        
        # store most important field so that it is not considered in next loop
        fields.remove(fields[imp_field])
        finished_fields.append(fields_to_test[np.argmin(np.array(pass_auc))])
        fields_to_test.remove(fields_to_test[np.argmin(np.array(pass_auc))])
        
    return ordered_fields
        
def feature_importance(model: tf.keras.Model, test_data: np.array, test_labels: np.array, iterations: int) -> None:

    '''
    Perform all feature importance
    args:
        model: DL model
        data: original data
        labels: labels for data
        iterations: number of iteration to then take an average of
    '''
    
    # set up results array for brieman test
    brieman = []
    
    # get brieman result for MSLP
    auc_arr = []
    for _ in range(iterations):
        loss, acc, tp, fp, fn, tn, recall, precision, auc, f1 = eval_model(model, brieman_feature_importance(test_data, [0]), test_labels)
        auc_arr.append(auc)
    brieman.append(["MSLP", np.mean(np.array(auc_arr))])
    
    # get brieman result for wind speed
    auc_arr = []
    for _ in range(iterations):
        loss, acc, tp, fp, fn, tn, recall, precision, auc, f1 = eval_model(model, brieman_feature_importance(test_data, [1]), test_labels)
        auc_arr.append(auc)
    brieman.append(["10m Wind Speed", np.mean(np.array(auc_arr))])

    # get brieman result for vorticity at 850hPa
    auc_arr = []
    for _ in range(iterations):
        loss, acc, tp, fp, fn, tn, recall, precision, auc, f1 = eval_model(model, brieman_feature_importance(test_data, [2]), test_labels)
        auc_arr.append(auc)
    brieman.append(["Vorticity at 850hPa", np.mean(np.array(auc_arr))])

    # get brieman result for vorticity at 700hPa
    auc_arr = []
    for _ in range(iterations):
        loss, acc, tp, fp, fn, tn, recall, precision, auc, f1 = eval_model(model, brieman_feature_importance(test_data, [3]), test_labels)
        auc_arr.append(auc)
    brieman.append(["Vorticitiy at 700hPa", np.mean(np.array(auc_arr))])
    
    # get brieman result for vorticity at 600hPa
    auc_arr = []
    for _ in range(iterations):
        loss, acc, tp, fp, fn, tn, recall, precision, auc, f1 = eval_model(model, brieman_feature_importance(test_data, [4]), test_labels)
        auc_arr.append(auc)
    brieman.append(["Vorticity at 600hPa", np.mean(np.array(auc_arr))])
    
    # calculate lakshmanan feature importance
    fields = ["MSLP", "10m Wind Speed", "Vorticity at 850hPa", "Vorticity at 700hPa", "Vorticity at 600hPa"]
    lak = lak_feature_importance(model, test_data, test_labels, fields, iterations)
    
    # parse brieman results into DataFrame
    brieman = pd.DataFrame(brieman)
    brieman = brieman.sort_values(by=[1], ascending=False)
    print(brieman)
    
    # parse lak results into DataFrame
    lak = pd.DataFrame(lak)
    lak = lak.sort_values(by=[1])
    print(lak)
    
    # plot
    plt.figure(figsize=(5, 5))

    vals = brieman[1]
    field_names = brieman[0]
    plt.subplot(211)
    plt.barh(np.array(list(range(0, 5))), vals)
    plt.yticks(range(5), field_names)

    vals = lak[1]
    field_names = lak[0]
    plt.subplot(212)
    plt.barh(np.array(list(range(0, 5))), vals)
    plt.yticks(range(5), field_names)

    plt.tight_layout()
    plt.savefig("feat_imp_whole_world.pdf")
    
def recall_by_cat(model: tf.keras.Model, path: str) -> tuple(np.array, np.array):

    '''
    Get the recall by category
    
    args:
        path: path to save location
    returns:
        data: cases from opened files as one numpy array; in the same order as files
        labels: label for each case in data
    '''
    
    #Get list of all files
    all_files=list(sorted(glob.glob(os.path.join(path, "*.npz"))))
    
    cat_recall = []
    for cat in range(1, 6):
        
        files = []
        for file in all_files:
            name = file.split("/")[-1].replace(".npz", "")
            if "no" in name:
                continue
            cat_file = int(name.split("_")[0])
            if cat_file == cat:
                files.append(file)
    
        # Open pool for multiprocessing
        pool = mp.Pool(int(0.5*mp.cpu_count()))

        # Get cases
        results = list(tqdm.tqdm(pool.imap(get_dl_file, files), total=len(files)))

        # Close pool after all cases loaded
        pool.close()
    
        # Set up numpy array to hold all cases
        shape = results[0][0].shape
        data = np.zeros((len(results), shape[0], shape[1], shape[2]))
        labels = np.zeros(len(results))

        # Loop through all cases and place in numpy array holding all cases
        for i, res in enumerate(results):
            data[i], labels[i] = res
    
        loss, acc, tp, fp, fn, tn, recall, precision, auc, f1 = eval_model(model, data, labels)
        cat_recall.append([cat, recall*100])
        
    cat_recall = pd.DataFrame(cat_recall)
    cat_recall.columns = ["Category", "Recall"]
    print(cat_recall)
    
    # Return the list of files opened and their case data
    return data, labels

def positive_cases_by_cat(model: tf.keras.Model, data: np.array, files: list[str]) -> None:

    '''
    Get number of cases positively and negativly inferred split by max cat TC in the region
    args:
        model: DL model
        data: preprocessed data
        files: list of file paths that correspond to the data given
    '''
    
    # set up counters
    no = 0
    minus_5 = 0
    minus_4 = 0
    minus_3 = 0
    minus_2 = 0
    minus_1 = 0
    zero = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0

    # make inferences
    preds = model.predict(data)
    
    # for each inference, increment relevant counter
    for pred_i, pred in enumerate(preds):
        
        if pred > 0.5:
            
            name = files[pred_i].split("/")[-1]
            cat = name.split("_")[0]
            
            if "no" in cat:
                no += 1
            else:
                cat = int(cat)

                if cat == -5:
                    minus_5 += 1
                elif cat == -4:
                    minus_4 += 1
                elif cat == -3:
                    minus_3 += 1
                elif cat == -2:
                    minus_2 += 1
                elif cat == -1:
                    minus_1 += 1
                elif cat == 0:
                    zero += 1
                elif cat == 1:
                    one += 1
                elif cat == 2:
                    two += 1
                elif cat == 3:
                    three += 1
                elif cat == 4:
                    four += 1
                elif cat == 5:
                    five += 1
                
    # collate results
    results = []
    results.append(["No meteorological system", "Unknown", "Post-tropical systems", "Disturbances", "Subtropical systems", "Tropcial Depressions", "Tropcial Storms", "Category 1 TCs", "Category 2 TCs", "Category 3 TCs", "Category 4 TCs", "Category 5 TCs"])
    results.append([no, minus_5, minus_4, minus_3, minus_2, minus_1, zero, one, two, three, four, five])
    results = pd.DataFrame(results).T
    results.columns = ["IBTrACS Label", "# cases"]
    
    # print results
    print(results)
    
def recall_by_basin(wawp_model: tf.keras.Model, model: tf.keras.Model, data: np.array, labels: np.array, files: list[str]) -> None:

    '''
    Get recall by basin for the WAWP and global models
    args:
        wawp_model: TCDetect trained on data from the WAWP basins
        model: TCDetect trained on global data
        data: preprocessed data
        labels: labels for the corresponding data
        files: file path for the given data
    '''
    
    # specify regions
    regions = ["_0_60_20_100", "_0_60_100_180", "_0_60_180_260", "_0_60_260_340", "-60_0_20_100", "-60_0_100_180", "-60_0_180_260", "-60_0_260_340"]
    
    # set up results list
    results = []
    
    # for each region
    for region in regions:
        
        # get indices for data cases which have a positive label
        idxs = []
        for file_i, file in enumerate(files):
            name = file.split("/")[-1].replace(".npz", "")
            cat = name.split("_")[0]
            if cat == "no":
                continue
            cat = int(cat)
            if cat < 1:
                continue
            if region in file:
                idxs.append(file_i)
        
        # get number of cases which have a positive inference
        num_pos = np.sum(labels[idxs])
        
        # get recall from both models
        if num_pos > 0:
            wawp_preds = wawp_model.predict(data[idxs])
            wawp_pos = 0
            for pred in wawp_preds:
                if pred > 0.5:
                    wawp_pos += 1
            wawp_recall = wawp_pos / num_pos

            global_preds = model.predict(data[idxs])
            global_pos = 0
            for pred in global_preds:
                if pred > 0.5:
                    global_pos += 1
            global_recall = global_pos / num_pos

            results.append([region, num_pos, wawp_recall*100, wawp_pos, global_recall*100, global_pos])
        else:
            results.append([region, 0, 0, 0, 0, 0])
            
    # print results
    results = pd.DataFrame(results).T
    print(results)
    
def get_mean_data(wawp_data: tf.keras.Mode, whole_world_data: np.array) -> None:

    '''
    Get and plot mean state for WAWP and global data
    args:
        wawp_data: data from WAWP regions
        whole_world_data: data from all regions
    '''
    
    fig, axs = plt.subplots(5, 2, figsize=(10, 20))

    axs[0, 0].contourf(np.mean(wawp_data[:, :, :, 0], axis = 0))
    axs[1, 0].contourf(np.mean(wawp_data[:, :, :, 1], axis = 0))
    axs[2, 0].contourf(np.mean(wawp_data[:, :, :, 2], axis = 0))
    axs[3, 0].contourf(np.mean(wawp_data[:, :, :, 3], axis = 0))
    axs[4, 0].contourf(np.mean(wawp_data[:, :, :, 4], axis = 0))

    axs[0, 1].contourf(np.mean(whole_world_data[:, :, :, 0], axis = 0))
    axs[1, 1].contourf(np.mean(whole_world_data[:, :, :, 1], axis = 0))
    axs[2, 1].contourf(np.mean(whole_world_data[:, :, :, 2], axis = 0))
    axs[3, 1].contourf(np.mean(whole_world_data[:, :, :, 3], axis = 0))
    axs[4, 1].contourf(np.mean(whole_world_data[:, :, :, 4], axis = 0))

    axs[0, 0].axis('off')
    axs[1, 0].axis('off')
    axs[2, 0].axis('off')
    axs[3, 0].axis('off')
    axs[4, 0].axis('off')
    axs[0, 1].axis('off')
    axs[1, 1].axis('off')
    axs[2, 1].axis('off')
    axs[3, 1].axis('off')
    axs[4, 1].axis('off')

    axs[0, 0].set_title("Mean MSLP")
    axs[1, 0].set_title("Mean 10m Wind Speed")
    axs[2, 0].set_title("Mean 850 hPa Vorticity")
    axs[3, 0].set_title("Mean 700 hPa Vorticity")
    axs[4, 0].set_title("Mean 600 hPa Vorticity")

    plt.tight_layout()
    plt.savefig("data_diff.pdf")
    
def dataset_size() -> None:

    '''
    Plot graph of dataset size vs loss and AUC-PR
    '''
    
    perc = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    loss = np.array([0.2441, 0.2847, 0.2963, 0.3184, 0.2216, 0.2853, 0.2477, 0.2805, 0.2986, 0.2650])
    auc = np.array([0.7001, 0.6850, 0.6906, 0.7027, 0.6896, 0.6963, 0.7149, 0.6950, 0.7087, 0.7173])
    host = host_subplot(111)
    par = host.twinx()
    host.set_xlabel("Percentage of Data Used / %")
    host.set_ylabel("Test AUC-PR / %")
    par.set_ylabel("Test Loss")
    p1, = host.plot(perc, auc, label="AUC-PR")
    p2, = par.plot(perc, loss, label="Loss")
    leg = plt.legend(loc=7)
    host.yaxis.get_label().set_color(p1.get_color())
    leg.texts[0].set_color(p1.get_color())
    par.yaxis.get_label().set_color(p2.get_color())
    leg.texts[1].set_color(p2.get_color())
    plt.tight_layout()
    plt.savefig("whole_world_dataset_size.pdf")
