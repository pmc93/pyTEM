#%%
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def process_params(S, params, inv_dir, i, survey_list, confile, src_file, dest_dir):
    try:
        print(f"Process {i} started. Params: {params}")
        
        process_inv_dir = os.path.abspath(f"{inv_dir}_{i}")
        print(os.path.join(process_inv_dir, src_file))
        if not os.path.exists(inv_dir):
            print(f"Base directory {inv_dir} does not exist!")
            raise FileNotFoundError(f"Base directory {inv_dir} not found.")

        print(f"Attempting to create directory: {process_inv_dir} for process {i}")
        if os.path.exists(process_inv_dir):
            shutil.rmtree(process_inv_dir)
        else:
            shutil.copytree(inv_dir, process_inv_dir)
            print(f"Directory '{process_inv_dir}' created successfully for process {i}")
            
   
        print(f"Cleaning {process_inv_dir} for process {i}")
        S.clean_inv_dir(process_inv_dir, extension='.mod')
        S.clean_inv_dir(process_inv_dir, extension='.emo')
        S.clean_inv_dir(process_inv_dir, extension='.err')
        S.clean_inv_dir(process_inv_dir, extension='.log')
        print(f"Cleaning completed for process {i}")


        print(f"Setting up start model for process {i}")
        start_model = {'num_layers': 3}
        start_model['rho'] = [params[0]] * start_model['num_layers']
        start_model['phi_max'] = [params[1]] * start_model['num_layers']
        start_model['tau_peak'] = [params[2]] * start_model['num_layers']
        start_model['c'] = [params[3]] * start_model['num_layers']
        start_model['thk'] = [params[4], params[4] + 5]
        start_model['depth'] = np.cumsum(start_model['thk']).tolist()

        start_model['alpha_r'] = 0.7
        start_model['d_r'] = 10
        start_model['p'] = 0.5
        start_model['depth_c'] = -1

        inv_settings = {'num_iterations': 50,
                        'param_layout': 114,
                        'constraints': 2}

        model_path = os.path.join(process_inv_dir, "my_model.mod")
        print(f"Writing model file to: {model_path} for process {i}")
        S.write_mod_file(model_path, survey_list=survey_list,
                         inv_settings=inv_settings, start_model=start_model,
                         default_value=-1.0000e+00)
        print(f"Model file written for process {i}")


        print(f"Running Aarhus inversion for process {i}")
        S.runAarhusInv(model_path, process_inv_dir, confile=confile)
        print(f"Aarhus inversion completed for process {i}")

        print(f"Cleaning up process directory: {process_inv_dir} for process {i}")
        S.clean_inv_dir(process_inv_dir, extension='.exe')
        S.clean_inv_dir(process_inv_dir, extension='.tem')
        S.clean_inv_dir(process_inv_dir, extension='.pdf')
        print(f"Process {i} completed successfully.")

        print(f"Copying and renaming file for process {i}")
        copy_and_rename_file(os.path.join(process_inv_dir, src_file), dest_dir, index=i+1, padding=3)


    except Exception as e:
        print(f"[Process {i}] Error occurred: {e}")
        # Clean up if there's an error
        #if os.path.exists(process_inv_dir):
        #    shutil.rmtree(process_inv_dir)
        raise

def run_threaded(S, params_set, inv_dir, survey_list, confile, src_file, dest_dir, num_threads=4):
    print(f"Starting threaded execution with {num_threads} threads")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_params, S, params, inv_dir, i, survey_list, confile, src_file, dest_dir): i
                   for i, params in enumerate(params_set)}

        for future in as_completed(futures):
            i = futures[future]
            try:
                future.result()  # This will raise any exception that occurred in the task
                print(f"Task {i} completed successfully")
            except Exception as e:
                print(f"Error in task {i}: {e}")

    print("All tasks finished")