#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import subprocess
import time
import contextily as cx

def read_data_block(lines):
     
    data = {}

    header_line = lines[0].strip()
    if header_line.startswith('$'):
        data['filename'] = header_line[1:]

    header_info_line = lines[1].strip()
    header_info = {}
    header_info_parts = header_info_line.split(';')
    for part in header_info_parts:
        if '=' in part:
            key, value = part.split('=')
            try:
                header_info[key] = float(value) if '.' in value else int(value)
            except ValueError:
                header_info[key] = value
    data['header_info'] = header_info

    # Read waveform data (assuming this is consistent across all blocks)
    data['tx_params'] = {}
    data['rx_params'] = {}
    data['tx_params']['SourceType'], data['rx_params']['Polar'] = map(int, lines[2].strip().split())

    if data['tx_params']['SourceType'] != 72:
        print("Error: Waveform type is not supported.")
        return

    center_coords = list(map(float, lines[3].strip().split()))
    data['tx_params']['CenterCoords'] = center_coords[0:3]
    data['rx_params']['CenterCoords'] = center_coords[3:6]

    data['tx_params']['NumSegments'], data['tx_params']['Area'] = map(float, lines[4].strip().split())
    data['tx_params']['NumSegments'] = int(data['tx_params']['NumSegments'])

    data['tx_params']['Coords'] = []
    for line in lines[5:5+data['tx_params']['NumSegments']]:
        data['tx_params']['Coords'].append(list(map(float, line.strip().split())))

    remaining_lines = lines[5+data['tx_params']['NumSegments']:]

    data['data_transform'] = list(map(int, remaining_lines[0].strip().split()))
    data['waveform_params'] = {}
    data['waveform_params']['Type'], data['waveform_params']['Number'] = list(map(int, remaining_lines[1].strip().split()))

    if data['waveform_params']['Type'] == 3:
        waveform_line = remaining_lines[2].strip().split()
        data['waveform_params']['NWaveformSegments'] = int(waveform_line[0])
        data['waveform_params']['WaveformDefinition'] = list(map(float, waveform_line[1:]))
    else:
        print("Error: Waveform type is not supported.")
        return

    data['filters'] = {}
    data['filters']['NFilters'] = int(remaining_lines[3].strip().split()[0])
    data['front_gate'] = {}
    data['front_gate']['ModelFrontGate'] = int(remaining_lines[3].strip().split()[1])
    data['tx_params']['DampFactor'] = float(remaining_lines[3].strip().split()[1])

    if data['filters']['NFilters'] != 1:
        print("Error: Code only supports 1 Filter.")
        return

    data['filters']['lowpass'] = {}
    data['filters']['lowpass']['NCutoff'] = int(remaining_lines[4].strip().split()[0])
    if data['filters']['lowpass']['NCutoff'] != 0:
        data['filters']['lowpass']['FOrder'] = int(remaining_lines[4].strip().split()[1])
        data['filters']['lowpass']['FCutoff'] = float(remaining_lines[4].strip().split()[2])

    data['filters']['highpass'] = {}
    data['filters']['highpass']['NCutoff'] = int(remaining_lines[5].strip().split()[0])
    if data['filters']['highpass']['NCutoff'] != 0:
        data['filters']['highpass']['FOrder'] = int(remaining_lines[5].strip().split()[1])
        data['filters']['highpass']['FCutoff'] = float(remaining_lines[5].strip().split()[2])

    if data['front_gate']['ModelFrontGate'] == 1:
        data['front_gate']['time'] = float(remaining_lines[6].strip().split()[0])

    data['front_gate']['lowpass'] = {}
    data['front_gate']['lowpass']['NCutoff'] = int(remaining_lines[7].strip().split()[0])
    if data['front_gate']['lowpass']['NCutoff'] != 0:
        data['front_gate']['lowpass']['FOrder'] = float(remaining_lines[7].strip().split()[1])
        data['front_gate']['lowpass']['FCutoff'] = float(remaining_lines[7].strip().split()[2])

    data['front_gate']['highpass'] = {}
    data['front_gate']['highpass']['NCutoff'] = int(remaining_lines[8].strip().split()[0])
    if data['front_gate']['highpass']['NCutoff'] != 0:
        data['front_gate']['highpass']['FOrder'] = int(remaining_lines[8].strip().split()[1])
        data['front_gate']['highpass']['FCutoff'] = float(remaining_lines[8].strip().split()[2])

    tem_data_lines = remaining_lines[9:]

    data['tem_data'] = parse_tem_data(tem_data_lines)

    return data

def parse_tem_data(lines):
    # Prepare list for DataFrame construction
    records = []
    
    for line in lines:
        # Skip lines that start with '%' which seem to be comments
        if line.startswith('%'):
            continue
        
        # Check if the line indicates the end of the data block
        if line.startswith('$'):
            break
        
        # Split the line into parts
        parts = line.split()
        if len(parts) < 5:
            print("Error: Data line does not have enough elements.")
            continue
        
        try:
            # Parse each line into the expected format
            gate_center = float(parts[0])
            signal = float(parts[1])
            err = float(parts[2])
            gate_open = int(parts[3])
            gate_close = int(parts[4])
            
            # Append the record to the list
            records.append([gate_center, signal, err, gate_open, gate_close])
        except ValueError as e:
            print(f"Error parsing line '{line}': {e}")
            continue

    # Create and return DataFrame
    df = pd.DataFrame(records, columns=['GateCenter', 'Signal', 'Error', 'GateOpen', 'GateClose'])
    
    return df
#%%

def read_tem_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    survey = read_data_block(lines)

    return survey

def read_da2_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    survey_list = []

    for i, line in enumerate(lines):
            stripped_line = line.strip()
            # Check if the line starts a new block
            if stripped_line.startswith('$'):
                survey_list.append(read_data_block(lines[i:]))

    return survey_list
#%%

def tem_files_to_da2(tem_files, output_file):
    with open(output_file, 'w') as outfile:
        for file in tem_files:
            with open(file, 'r') as infile:
                # Write the filename as a new line
                outfile.write(f"${os.path.basename(file)}\n")
                # Append the contents of the file
                outfile.write(infile.read())

def write_tem_files(survey_list):

    for survey in survey_list:

        write_tem_file(survey)

def write_tem_file(data):
    with open(data['filename'], 'w') as f:
        # Write the header line
        #if 'filename' in data:
        #    f.write(f"${data['filename']}\n")
        
        # Write the header info
        if 'header_info' in data:
            header_info = data['header_info']
            header_info_str = ';'.join(f"{k}={v}" for k, v in header_info.items())
            f.write(f"{header_info_str}\n")
        
        # Write waveform data
        tx_params = data.get('tx_params', {})
        rx_params = data.get('rx_params', {})
        if 'SourceType' in tx_params and 'Polar' in rx_params:
            f.write(f"{tx_params['SourceType']} {rx_params['Polar']}\n")
        
        if 'CenterCoords' in tx_params and 'CenterCoords' in rx_params:
            center_coords = tx_params['CenterCoords'] + rx_params['CenterCoords']
            f.write(" ".join(f"{x:.3e}" for x in center_coords) + "\n")
        
        if 'NumSegments' in tx_params and 'Area' in tx_params:
            f.write(f"{tx_params['NumSegments']} {tx_params['Area']:.3e}\n")
        
        if 'Coords' in tx_params:
            for coord in tx_params['Coords']:
                f.write(" ".join(f"{x:.3e}" for x in coord) + "\n")
        
        # Write the remaining lines
        if 'data_transform' in data:
            f.write(" ".join(str(x) for x in data['data_transform']) + "\n")
        
        if 'waveform_params' in data:
            waveform_params = data['waveform_params']
            f.write(f"{waveform_params['Type']} {waveform_params['Number']}\n")
            
            if waveform_params['Type'] == 3:
                wf_def = [str(waveform_params['NWaveformSegments'])] + [f"{x:.6e}" for x in waveform_params['WaveformDefinition']]
                f.write(" ".join(wf_def) + "\n")
        
        if 'filters' in data:
            filters = data['filters']
            if 'NFilters' in filters:
                f.write(f"{filters['NFilters']} {data['front_gate'].get('ModelFrontGate', 1)}\n")
            
            if 'lowpass' in filters:
                lp = filters['lowpass']
                f.write(f"{lp['NCutoff']} {lp.get('FOrder', 0)} {lp.get('FCutoff', 0.0):.6e}\n")
            
            if 'highpass' in filters:
                hp = filters['highpass']
                f.write(f"{hp['NCutoff']} {hp.get('FOrder', 0)} {hp.get('FCutoff', 0.0):.6e}\n")
        
        if 'front_gate' in data:
            fg = data['front_gate']
            if fg.get('ModelFrontGate', 0) == 1:
                f.write(f"{fg.get('time', 0.0):.6e}\n")
            
            if 'lowpass' in fg:
                lp = fg['lowpass']
                f.write(f"{lp['NCutoff']} {lp.get('FOrder', 0.0)} {lp.get('FCutoff', 0.0):.3e}\n")
            
            if 'highpass' in fg:
                hp = fg['highpass']
                f.write(f"{hp['NCutoff']} {hp.get('FOrder', 0)} {hp.get('FCutoff', 0.0):.3e}\n")
        
        # Write tem_data
        if 'tem_data' in data:
            for index, row in data['tem_data'].iterrows():
                f.write("".join(f"{row['GateCenter']:.6e} {row['Signal']:.6e} {row['Error']:.6e} {int(row['GateOpen']):d} {int(row['GateClose']):d}\n"))



def write_model_block(f, start_model, parameter, default_value, alpha=9e9, last_model=False):
    param1 = 1
    param2 = 1
    param3 = -1
    param4 = -1

    if last_model == False:

        if isinstance(alpha, str):
            alpha = start_model[alpha]

        if parameter != 'thk' and parameter != 'depth':
            for i, value in enumerate(start_model[parameter]):
                if i == start_model['num_layers']-1:
                    f.write(f"{value: .4e} {param3: .4e} {'' :>11} {param1:11d} {param2:11d} {alpha: .4e}\n")
                else:
                    f.write(f"{value: .4e} {param3: .4e} {default_value: .4e} {param1:11d} {param2:11d} {alpha: .4e}\n")

        elif parameter == 'thk':
            for i, value in enumerate(start_model[parameter]):
                if i == start_model['num_layers']-2:
                    f.write(f"{value: .4e} {param4: .4e} {'' :>11} {param1:11d} {param2:11d} {default_value: .4e}\n")
                else:
                    f.write(f"{value: .4e} {param4: .4e} {default_value: .4e} {param1:11d} {param2:11d} {default_value: .4e}\n")

        else:
            for i, value in enumerate(start_model[parameter]):
                f.write(f"{value: .4e} {param3: .4e} {'' :>11} {param1:11d} {param2:11d} {default_value: .4e}\n")

    else:

        param1 = 0

        if parameter != 'thk' and parameter != 'depth':
            for i, value in enumerate(start_model[parameter]):
                if i == start_model['num_layers']-1:
                    f.write(f"{value: .4e} {param3: .4e} {'' :>11} {param1:11d}\n")
                else:
                    f.write(f"{value: .4e} {param3: .4e} {default_value: .4e} {param1:11d}\n")

        elif parameter == 'thk':
            for i, value in enumerate(start_model[parameter]):
                if i == start_model['num_layers']-2:
                    f.write(f"{value: .4e} {param4: .4e} {'' :>11} {param1:11d}\n")
                else:
                    f.write(f"{value: .4e} {param4: .4e} {default_value: .4e} {param1:11d}\n")

        else:
            for i, value in enumerate(start_model[parameter]):
                f.write(f"{value: .4e} {param3: .4e} {'' :>11} {param1:11d}\n")


def write_mo2_file(filename, survey_list, inv_settings, start_model, default_value):
    with open(filename, 'w') as f:
        # Write the header
        f.write("My hyggelig model file\n")
        f.write(f"{len(survey_list)} {inv_settings['constraints']}\n")
        
        # Write the file entries
        for i, survey in enumerate(survey_list):
            f.write(f"{i+1} {inv_settings['param_layout']} {survey['filename']}\n")
        
        # Write the number of inversions
        f.write(f"{inv_settings['num_iterations']}\n")
        
        # Write the layer information
        for i, survey in enumerate(survey_list):

            x = survey['header_info']['XUTM']
            y = survey['header_info']['YUTM']
            z = survey['header_info']['Elevation']

            f.write(f"{start_model['num_layers']:5d} {x: .1f} {y: .1f} {z: .1f}\n")

            if i != len(survey_list)-1:

                d = ((x - survey_list[i+1]['header_info']['XUTM']) ** 2 + (y - survey_list[i+1]['header_info']['YUTM']) ** 2) ** 0.5

            #alpha = (start_model['alpha_r'] - 1) * (d / start_model['d_r']) ** start_model['p'] + 1

            alpha = 9e9

            last_model = False

            if i == len(survey_list)-1:
                last_model = True

            write_model_block(f, start_model, parameter='rho', default_value=start_model['rho_v'], alpha=alpha, last_model=last_model)
            write_model_block(f, start_model, parameter='phi_max', default_value=start_model['phi_max_v'], alpha=alpha, last_model=last_model)
            write_model_block(f, start_model, parameter='tau_peak', default_value=start_model['tau_peak_v'], alpha=alpha, last_model=last_model)
            write_model_block(f, start_model, parameter='c', default_value=start_model['c_v'], alpha=alpha, last_model=last_model)
            write_model_block(f, start_model, parameter='thk', default_value=-1, alpha=alpha, last_model=last_model)
            write_model_block(f, start_model, parameter='depth', default_value=default_value, alpha='depth_c', last_model=last_model)
        
def rename_surveys(survey_list, site_name):

    num_digits = len(str(len(survey_list)))
    
    for i, survey in enumerate(survey_list):
        
        survey['filename'] = f"{site_name}_{str(i+1).zfill(num_digits)}.{'tem'}"



def plot_column_from_surveys(survey_list):
    # Initialize a list to store the column data from all surveys
    all_data = []

    # Loop through each survey in the survey_list
    for survey in survey_list:
        # Extract the "Signal" data from the DataFrame within the survey dictionary
        data = survey['tem_data']['Signal']
        all_data.append(data)
    
    # Combine all the data into a single DataFrame
    combined_data = pd.concat(all_data, axis=1)

    # Set the survey indices as columns
    combined_data.columns = range(1, len(survey_list) + 1)

    # Transpose the data so that each survey's data is a row
    transposed_data = combined_data.transpose()

    # Plot the transposed data
    plt.figure(figsize=(10, 6))

    # Plot positive values in red with lines

    for i in transposed_data.columns:
        clean_data = transposed_data[i].dropna()
        
        # Plot the data with a dashed line
        plt.plot(clean_data.index, np.abs(clean_data), 'k--')



    for i in transposed_data.columns:
        positive_values = transposed_data[i][transposed_data[i] > 0]
        if not positive_values.empty:
            plt.plot(positive_values.index, positive_values, 'ro')

    # Plot negative values in blue with lines
    for i in transposed_data.columns:
        negative_values = transposed_data[i][transposed_data[i] < 0]
        if not negative_values.empty:
            plt.plot(negative_values.index, np.abs(negative_values), 'bo')

    plt.xticks(ticks=np.arange(1, len(survey_list) + 1), labels=np.arange(1, len(survey_list) + 1))

    plt.xlabel("Survey Index")
    plt.ylabel("Signal")
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    

def smart_figsize_from_data(x, y, base_height=5):
    """Adjust figsize based on the aspect ratio of the data range."""
    aspect_ratio = (max(x) - min(x)) / (max(y) - min(y))
    width = base_height * aspect_ratio
    return (width, base_height)

def plot_survey_locs(survey_list, ax=None, basemap=False, epsg=None, text_color='w'):

    if ax is None:
        fig, ax = plt.subplots()  

    x_values = [survey['header_info']['XUTM'] for survey in survey_list]
    y_values = [survey['header_info']['YUTM'] for survey in survey_list]

    cols = range(len(x_values))

    # Create a scatter plot
    sc = ax.scatter(x_values, y_values, c=cols, cmap='turbo')

    for i in range(len(survey_list)):
        ax.text(x_values[i], y_values[i], str(i), fontsize=9, 
                ha='right', color= text_color)

    if basemap:
        if epsg is None:
            print('Please define a epsg code, e.g., epsg:32629')
        else:
            cx.add_basemap(ax=ax, crs=epsg, source=cx.providers.Esri.WorldImagery, attribution=False)

    # Add labels and title
    ax.set_xlabel('UTMX [m]')
    ax.set_ylabel('UTMY [m]')

    ax.set_aspect(1)

def reorder_survey_points(survey_list, start_index=0):
    # Make a copy of the survey_list to avoid modifying the original list
    survey_list_copy = survey_list.copy()
    
    # Ensure start_index is within bounds
    if start_index < 0 or start_index >= len(survey_list_copy):
        raise ValueError("start_index is out of range")
    
    # Start with the specified point
    reordered_list = [survey_list_copy.pop(start_index)]
    
    while survey_list_copy:
        last_point = reordered_list[-1]
        last_x = last_point['header_info']['XUTM']
        last_y = last_point['header_info']['YUTM']
        
        # Find the closest point to the last point in reordered_list
        distances = [
            np.sqrt((survey['header_info']['XUTM'] - last_x) ** 2 +
                    (survey['header_info']['YUTM'] - last_y) ** 2)
            for survey in survey_list_copy
        ]
        # Find the index of the closest point
        closest_idx = np.argmin(distances)
        # Append the closest point to reordered_list and remove it from survey_list_copy
        reordered_list.append(survey_list_copy.pop(closest_idx))
    
    return reordered_list

def read_emo_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Define patterns for extraction
    n_iterations_pattern = re.compile(r'^N iterations \(NI, 0 -> analysis and forward resp.\)\s+(\d+)', re.MULTILINE)
    n_data_sets_pattern = re.compile(r'^N data sets \(NDS\)\s+(\d+)', re.MULTILINE)

    def extract_value(pattern, content):
        match = pattern.search(content)
        if match:
            return int(match.group(1))
        return None

    # Extract the number of iterations and data sets
    n_iterations = extract_value(n_iterations_pattern, content)
    n_data_sets = extract_value(n_data_sets_pattern, content)

    with open(filename, 'r') as file:
        lines = file.readlines()
    
    i = 0
    num_lines = len(lines)

    inv_model_list = []
    param_count = 0
    fwd_count = 0

    n_data = []

    while i < num_lines:
        line = lines[i].strip()

        if line.startswith('N data in each data set (1..NDS)'):
            for j in range(1, n_data_sets+1):
                n_data.append(int(lines[i+j].strip().split()[0]))
                
        if line.startswith('Model #'):
            # Collect the model info if it's not just an integer
            if i + 1 < num_lines and not lines[i + 1].strip().isdigit():
                inv_model = {}
                inv_model_list.append(inv_model)
                param_block = True
                line = lines[i + 1].strip().split()
                #inv_model_list[param_count]['info'] = {}
                inv_model_list[param_count]['x'] = float(line[2])
                inv_model_list[param_count]['y'] = float(line[3])
                inv_model_list[param_count]['elevation'] = float(line[4]) 

            else:
                param_block = False               

            # Move to the next line where parameters start
            i += 2

            # Check if the next line starts with "Parameters"
            if i < num_lines and lines[i].strip().startswith('Parameters (0..NIte)'):
                i += 2  # Move to the first parameter line

                # Collect parameter lines
                inv_model_list[param_count]['parameters'] = []
                for j in range(n_iterations):
                    if i < num_lines:
                        inv_model_list[param_count]['parameters'].append(lines[i].strip())
                        i += 1
                    else:
                        break
                processed_data = [line.split()[1:] for line in inv_model_list[param_count]['parameters']]  # Exclude the index

                # Convert to DataFrame
                df = pd.DataFrame(processed_data)

                inv_model_list[param_count]['parameters'] = df.astype(float)
                
            if param_block == True:
                param_count += 1

        elif line.startswith('Data type ID') and not line.startswith('Data type ID(s)'):
            #print(f"\nFound 'Data type ID' at line {i}. Reading the next {n_data[fwd_count]} lines:")
            i += 3
            inv_model_list[fwd_count]['fwd_response'] = []
            
            # Print the specified number of lines after the "Data type ID"
            for j in range(n_data[fwd_count]):
                if i < num_lines:
                    inv_model_list[fwd_count]['fwd_response'].append(lines[i].strip())
                    i += 1
                else:
                    break  # Stop if there are not enough lines left to read
            
            processed_data = [line.split()[0:] for line in inv_model_list[fwd_count]['fwd_response']]  # Exclude the index

            df = pd.DataFrame(processed_data)

            inv_model_list[fwd_count]['fwd_response'] = df.astype(float)

            fwd_count += 1
        else:
            i += 1
    
    return inv_model_list


def plot_response(inv_models, idx, ax=None, exclude_gates=None):

    if ax is None:
        fig, ax = plt.subplots()

    df = inv_models[idx]['fwd_response'].copy()

    t = df.iloc[:, 0]  # Time data
    obs = df.iloc[:, 1]  # Observed data
    fwd = df.iloc[:, 6]  # Forward model data
    err = df.iloc[:, 2] # Err

    if exclude_gates is not None:
        df = df.drop(exclude_gates)

    t_filt = df.iloc[:, 0]  # Time data
    obs_filt = df.iloc[:, 1]  # Observed data
    fwd_filt = df.iloc[:, 6]  # Forward model data
    err_filt = df.iloc[:, 2] # Err

    misfit = np.sum(np.abs(obs_filt - fwd_filt) / (np.abs(fwd_filt) * err_filt)) / len(obs_filt)

    print(misfit)

    # Separate positive and negative observed data
    obs_positive = np.where(obs_filt > 0, obs_filt, np.nan)  # Keep positive values, NaN for others
    obs_negative = np.where(obs_filt < 0, obs_filt, np.nan)  # Keep negative values, NaN for others

    ax.loglog(t, np.abs(obs), marker='x', lw=0, color='grey')

    # Plot positive observed data in blue and negative in red
    ax.loglog(t_filt, obs_positive, marker='x', lw=0, color='b')
    ax.loglog(t_filt, -obs_negative, marker='x', lw=0, color='r')

    ax.loglog(t, np.abs(fwd), c = 'k')

    ax.set_title(f"Misfit= {misfit:.2f}")

def runAarhusInv(mod_file_name, inv_dir, confile):
    t = time.time()  # Start time

    os.chdir(inv_dir)

    try:
        result = subprocess.run(
            ['AarhusInv64_v10.1.exe', mod_file_name, confile], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            cwd=inv_dir,  # Set the working directory
            check=True  # Raises an exception if the process fails
        )
        
        # Output the result if necessary
        #print(result.stdout.decode())  # Prints the output of the subprocess if needed
        elapsed = time.time() - t  # End time
        print(f"AarhusInv finished in  {np.round(elapsed, 2)} seconds.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"Error output: {e.stderr.decode()}")

def clean_inv_dir(inv_dir, extension):
    """
    Deletes all files with a specific extension in the specified folder.

    Parameters:
    folder_path (str): The path to the folder from which to delete files.
    extension (str): The extension of the files to delete (e.g., '.txt').
    """
    try:
        # Check if the folder exists
        if os.path.exists(inv_dir):
            # Iterate through all the files in the folder
            for filename in os.listdir(inv_dir):
                file_path = os.path.join(inv_dir, filename)
                # Check if the file has the specified extension and is a file
                if filename.endswith(extension) and os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            print(f"All files with extension '{extension}' deleted successfully.")
        else:
            print(f"The folder {inv_dir} does not exist.")
    except Exception as e:
        print(f'Error occurred: {e}')


def remove_gates(survey_list, exclude_gates):


    for survey in survey_list:
        df = survey['tem_data'].copy()
        try:
            df = df.drop(exclude_gates)
            survey['tem_data'] = df
        except Exception as e:
            print(f'Failed to delete {exclude_gates}. Reason: {e}')

    return survey_list

def change_error(survey_list):


    for survey in survey_list:
        df = survey['tem_data'].copy()
        df.iloc[:,2] = df.iloc[:,2]*2
        survey = df

    return survey_list

def select_surveys(survey_list, survey_idx):

    return [survey_list[i] for i in survey_idx if i < len(survey_list)]
