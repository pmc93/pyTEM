#%%

import matplotlib.pyplot as plt

import re

import contextily as cx
import numpy as np
import os
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import itertools
import os

import os
import shutil
import stat


import survey as S
from plot_tem import *


%matplotlib qt

#%%Test


locs = ['Badioula', 'Bembou', 'Kalando', 'Saraya South', 'Saraya Northeast', 'Saraya Northwest', 'Majiera']

loc = locs[0]
if loc == 'Badioula':
    survey_list1 = S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Badioula_1102.da2')
    survey_list2 = S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Badioula_1202_1.da2')
    survey_list3 = S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Badioula_1202_2.da2')

    survey_list = survey_list1 + survey_list2 + survey_list3

    T1 = [42, 45, 44, 47, 48, 50, 49, 52, 
            51, 54, 53, 0, 56, 1, 55, 18, 2, 58, 
             3, 57, 4, 5, 6, 7, 8, 9, 10, 11, 
             12, 13, 14, 15, 16, 17]

    #46 and 59 removed from T1

    T2 =[43, 40, 41, 38, 39, 37, 34, 35, 36, 
             32, 33, 56, 1, 30, 31, 28, 2, 29, 
             26, 27, 24, 25, 22, 23, 21, 19, 20]

    Ts = [T1, T2]


fig, axs = plt.subplots(1,2)

S.plot_survey_locs(survey_list, ax = axs[0], basemap=True, epsg='epsg:32629')

survey_list_T = []

for i, T in enumerate(Ts):
    survey_list_T.append(S.select_surveys(survey_list, T))
    survey_list_T[i] = S.reorder_survey_points(survey_list=survey_list_T[i], start_index=0)
    S.rename_surveys(survey_list_T[i], site_name=loc)
    S.plot_survey_locs(survey_list_T[i], ax=axs[1], basemap=True, epsg='epsg:32629')

    x = []
    y = []

    for j in range(len(survey_list_T[i])):
        x.append(survey_list_T[i][j]['header_info']['XUTM'])
        y.append(survey_list_T[i][j]['header_info']['YUTM'])

    axs[1].plot(x, y, label = F'Transect {i+1}')

axs[1].legend()

fig.tight_layout()

plt.savefig(f'{loc}_map.png')
#%%
k = 0

inv_dir = r"C:\Users\pamcl\AarhusInv\inv_dir"
os.chdir(inv_dir)

S.clean_inv_dir(inv_dir, extension='.tem')
S.clean_inv_dir(inv_dir, extension='.emo')

S.write_tem_files(survey_list_T[k])

start_rhos = [20, 40, 100, 200, 400]
phi_maxs = [5, 30, 100]
tau_peaks = [1e-6, 1e-4, 1e-2]
cs = [0.2, 0.5, 0.8]
thk1s = [5, 10, 15, 20]

params_set = list(itertools.product(start_rhos, phi_maxs, tau_peaks, cs, thk1s))


#%%

confile = "SarayaConfile.con"
src_file = 'my_model.emo'
dest_dir = r"c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\inv_results/"+loc+"_T"+str(k+1)

#%%
def delete_folders(directory_path, start_index=1):
    """
    Deletes folders within a directory starting from a specified index, modifying permissions if necessary.

    Parameters:
    - directory_path (str): The path to the parent directory.
    - start_index (int): The index from which to start deleting folders (default is 1).
    """
    # List only folders (directories) in the specified path
    folders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]
    
    # Ensure the start_index is within the range of available folders
    if start_index >= len(folders):
        print(f"Error: Start index {start_index} is out of range.")
        return

    # Loop through folders from the start_index onward
    for i in range(start_index, len(folders)):
        folder_to_delete = os.path.join(directory_path, folders[i])
        
        # Change permissions for all files in the directory to ensure deletion
        for root, dirs, files in os.walk(folder_to_delete):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.chmod(file_path, stat.S_IWRITE)  # Grant write permission
                except Exception as e:
                    print(f"Could not change permissions for {file_path}: {e}")

        # Attempt to delete the folder
        try:
            shutil.rmtree(folder_to_delete)
            print(f"Folder '{folders[i]}' has been deleted successfully.")
        except Exception as e:
            print(f"Failed to delete folder '{folders[i]}': {e}")

# Usage
directory_path = r'C:\Users\pamcl\AarhusInv'
delete_folders(directory_path)
#%%
import saraya_parallel as p
importlib.reload(p)

p.run_threaded(S, params_set, inv_dir, survey_list_T[0], confile, src_file, dest_dir, num_threads=20)

dest_dir = r"C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\inv_results\Badioula_T1"

emo_files = [file for file in os.listdir(dest_dir) if file.endswith('.emo')]
n_soundings = len(S.read_emo_file(os.path.join(dest_dir, emo_files[0])))


#%%

api_key = "AIzaSyCsFOM4HgQKKOHjRAqxudktkKEi-9nbfzw"  # Replace with your Google Maps API key

import utm

# Function to convert UTM to Latitude and Longitude
def utm_to_latlong(easting, northing, zone_number, zone_letter):
    lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    return lat, lon

# Function to get elevation from Google Maps Elevation API
def get_elevation(latitude, longitude, api_key):
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "locations": f"{latitude},{longitude}",
        "key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        result = response.json()
        if result["status"] == "OK":
            elevation = result["results"][0]["elevation"]
            return elevation
        else:
            print("Error:", result["status"])
    else:
        print(f"HTTP Error: {response.status_code}")
    return None

emo_dir = r"C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\inv_results\Badioula_T1"
elev_new = []
models = S.read_emo_file(os.path.join(emo_dir, emo_files[0]))

for model in models:
    x=(model['x'])
    y=(model['y'])

    zone_number=29; zone_letter='N' 
    lat, lon = utm_to_latlong(x, y, zone_number, zone_letter)
    elev_new.append(get_elevation(lat, lon, api_key))

#%%
fig, axs = plt.subplots(4, 3, sharex=True)

idx = [0, 10, 30, 33, 25, 20, 12, 8, 5, 22, 31, 26]

axs = axs.flatten()
for j, idx in enumerate(idx):

    ax = axs[j]

    model_id = accepted_models[idx]

    models = S.read_emo_file(os.path.join(dir, emo_files[model_id]))

    x = []
    y = []
    profiler_list = []


    for i, model in enumerate(models):
        
        x.append(model['x'])
        y.append(model['y'])

        profiler = {}
        
        profiler['x'] = model['x']
        profiler['y'] = model['y']

        profiler['elevation'] = elev_new[i]
        profiler['rhos'] = model['parameters'].values[-1, 0:3]
        profiler['phi_maxs'] = model['parameters'].values[-1, 3:6]
        profiler['tau_peaks'] = model['parameters'].values[-1, 6:9]
        profiler['cs'] = model['parameters'].values[-1, 9:12]
        profiler['thk'] = model['parameters'].values[-1, 12:14]
        profiler['bot_depths'] = np.cumsum(profiler['thk'])
        profiler['bot_depths'] = np.append(profiler['bot_depths'], profiler['bot_depths'][-1]*2) 
        profiler['top_depths'] = np.insert(profiler['bot_depths'][:-1], 0, 0) 
        profiler['doi'] = 100
        profiler['n_layers'] = len(profiler['top_depths'])
        profiler_list.append(profiler)

    xi, yi, dists = interpolate_points(np.array(x), np.array(y), distance=1)