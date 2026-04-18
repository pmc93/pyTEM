#%%
import shutil
import os
import numpy as np

# functions for doing aarhusinv in parallel

def copy_and_rename_file(src_file, dest_dir, index, padding=3):
    # Check if destination directory exists, if not, create it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Extract the file name and extension from the source file
    file_name, file_extension = os.path.splitext(os.path.basename(src_file))

    # Generate new file name with zfill (padding)
    new_file_name = f"{file_name}{str(index).zfill(padding)}{file_extension}"

    # Construct the full path for the destination file with the new name
    dest_file = os.path.join(dest_dir, new_file_name)

    # Copy the file to the new location with the new name
    shutil.copy(src_file, dest_file)

    print(f'{src_file} copied and renamed to {dest_file}')


def run_mpa_inv(S, inv_dir, confile, mod_file, start_model, survey_list):

    os.chdir(inv_dir)

    S.clean_inv_dir(inv_dir, extension='.mod')
    S.clean_inv_dir(inv_dir, extension='.emo')
    S.clean_inv_dir(inv_dir, extension='.err')
    S.clean_inv_dir(inv_dir, extension='.log')
        
    inv_settings = {'num_iterations': 50,
                    'param_layout': 114,
                    'constraints': 2}

    model_path = os.path.join(inv_dir, mod_file)
    print(f"Writing model file to: {model_path}")
    S.write_mo2_file(model_path, survey_list=survey_list,
                     inv_settings=inv_settings, start_model=start_model,
                     default_value=-1.0000e+00)
    
    S.runAarhusInv(model_path, inv_dir, confile=confile)

def run_rho_inv(S, inv_dir, confile, mod_file, start_model, survey_list):

    os.chdir(inv_dir)

    S.clean_inv_dir(inv_dir, extension='.mod')
    S.clean_inv_dir(inv_dir, extension='.emo')
    S.clean_inv_dir(inv_dir, extension='.err')
    S.clean_inv_dir(inv_dir, extension='.log')

    n_layers = 30

    existing_depths = start_model['depth']  # Depths from the existing model
    existing_rho = start_model['rho'] # Resistivities from the existing model

    start_model = {'num_layers': n_layers}
    start_model['rho'] = np.ones((n_layers)) * 100.
    start_model['depth'] = np.array([1., 2.088, 3.272, 4.56, 5.962, 7.487, 9.147,
                                     10.953, 12.918, 15.056, 17.382, 19.913, 22.667, 
                                     25.664, 28.925, 32.473, 36.334, 40.535, 45.106, 
                                     50.08 , 55.492, 61.381, 67.789, 74.762, 82.349, 
                                     90.604, 99.587, 109.361, 120.001]) * 2
    
    start_model['thk'] = np.diff(np.insert(start_model['depth'], 0, 0))
        
    inv_settings = {'num_iterations': 50,
                    'param_layout': 1,
                    'constraints': 1}

    model_path = os.path.join(inv_dir, mod_file)
    print(f"Writing model file to: {model_path}")
    S.write_mo2_file(model_path, survey_list=survey_list,
                     inv_settings=inv_settings, start_model=start_model,
                     default_value=-1.0000e+00)
    
    S.runAarhusInv(model_path, inv_dir, confile=confile)

def run_rho_fwd(S, inv_dir, confile, mod_file, start_model, survey_list):

    os.chdir(inv_dir)

    S.clean_inv_dir(inv_dir, extension='.mod')
    S.clean_inv_dir(inv_dir, extension='.emo')
    S.clean_inv_dir(inv_dir, extension='.err')
    S.clean_inv_dir(inv_dir, extension='.log')
        
    inv_settings = {'num_iterations': -1,
                    'param_layout': 1,
                    'constraints': 0}

    model_path = os.path.join(inv_dir, mod_file)
    print(f"Writing model file to: {model_path}")
    S.write_mo2_file(model_path, survey_list=survey_list,
                     inv_settings=inv_settings, start_model=start_model,
                     default_value=-1.0000e+00)
    
    S.runAarhusInv(model_path, inv_dir, confile=confile)
