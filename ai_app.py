#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ct
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from tkinter import filedialog, messagebox
import socket

import sys

sys.path.append(r'C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\Pytem/')
import survey as S

import inv_helper as ih

sys.path.append(r'C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\Pytem0')

from import_colors import getAarhusCols, getParulaCols
from plot_tem import *

matplotlib.use('Agg')

class PyTEMApp:
    def __init__(self):

        def check_connection():
            try:
                # Attempt to connect to Google's DNS server
                socket.create_connection(("8.8.8.8", 53), timeout=1)
                return True
            except OSError:
                return False
            
        self.connection = check_connection()

        # Initialize the CustomTkinter appearance
        ct.set_default_color_theme('blue')
        ct.set_appearance_mode('Dark')
        
        self.fwd_response = None
        self.model_params = None
        self.inv_dir = None

        # Initialize main variables
        self.n_layers = 2  # Number of layers
        self.canvas = None  # To store canvas for plotting
        self.model_type = "Resistivity"  # Default model type
        self.param_labels = ['Layer', 'Thickness\n[m]', 'Rho\n[Ohm.m]']
        self.n_params = len(self.param_labels) - 1
        self.survey_index = None

        # Set up the main Tkinter window
        self.root = ct.CTk()
        self.root.title("pyTEM")
        self.main_frame = ct.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Set up frames for plot, data, layers, and buttons
        self.setup_frames()

        # Run the Tkinter main loop
        self.root.mainloop()

    # Functions for defining app layout
    def distribute_buttons(self, frame, buttons):
        # Number of buttons determines number of columns
        num_buttons = len(buttons)

        # Configure the grid columns for equal weight
        for col in range(num_buttons):
            frame.grid_columnconfigure(col, weight=1)

        # Place each button in its respective column
        for col, button in enumerate(buttons):
            button.grid(row=0, column=col, padx=5, pady=5, sticky="ew")

    def setup_frames(self):
        # Plot frame
        self.plot_frame = ct.CTkFrame(self.main_frame, width=0)
        self.plot_frame.grid(row=0, column=1, rowspan=6, padx=5, pady=5, sticky="nsew")

        # Data frame
        self.data_frame = ct.CTkFrame(self.main_frame, width=400)
        self.data_frame.grid(row=0, column=0, padx=5, pady=10, sticky="ew")
        button1 = ct.CTkButton(self.data_frame, text="Load System Parameters", width=90, 
                               command=lambda: print('Load System Parameters'))
        button1.configure(state=ct.DISABLED)
        button2 = ct.CTkButton(self.data_frame, text="Load Data", 
                               command=self.open_da2, width=90)
        button3 = ct.CTkButton(self.data_frame, text="Plot Map", 
                               command=self.open_map_window, width=90)
        button3.configure(state=ct.DISABLED)
        self.survey_index_entry = ct.CTkEntry(self.data_frame, placeholder_text="Survey Index", width=90)
        self.survey_index_entry.configure(state=ct.DISABLED)
        button4 = ct.CTkButton(self.data_frame, text="Plot Sounding", command=self.plot_sounding, width=90)
        button4.configure(state=ct.DISABLED)
        button5 = ct.CTkButton(self.data_frame, text="Plot All", command=self.plot_all, width=90)
        button5.configure(state=ct.DISABLED)
        self.distribute_buttons(self.data_frame, [button1, button2, self.survey_index_entry, button3, button4, button5])
        self.data_frame_buttons = [button1, button2, button3, button4, button5]
        
        # Layer frame
        self.layer_frame = ct.CTkFrame(self.main_frame, width=200)
        self.layer_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        button1 = ct.CTkButton(self.layer_frame, text="Create Start Model", command=self.create_start_model)
        button1.configure(state=ct.DISABLED)
        button2 = ct.CTkButton(self.layer_frame, text="Add Layer", command=self.add_layer)
        button2.configure(state=ct.DISABLED)
        button3 = ct.CTkButton(self.layer_frame, text="Remove Layer", command=self.remove_layer)
        button3.configure(state=ct.DISABLED)
        #self.model_type_combo = ct.CTkComboBox(self.layer_frame, values=["Resistivity", "Pelton", "Double Pelton", "MPA"])
        self.model_type_combo = ct.CTkComboBox(self.layer_frame, values=["Resistivity", "MPA"])
        self.model_type_combo.set("Resistivity")  # Set the default value
        self.model_type_combo.configure(state=ct.DISABLED)
        self.distribute_buttons(self.layer_frame, [button1, button2, button3, self.model_type_combo])
        self.layer_frame_buttons = [button1, button2, button3, button4]

        # Scrollable layer frame
        self.layer_scrollable_frame = ct.CTkScrollableFrame(self.layer_frame, width=600)
        self.layer_scrollable_frame.grid(row=1, column=0, sticky="nsew", columnspan=4)

        # Add button to calculate forward model
        self.model_frame = ct.CTkFrame(self.main_frame, width=400)
        self.model_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        button1 = ct.CTkButton(self.model_frame, text="Calculate Forward Model", 
                               command=self.run_fwd)
        button1.configure(state=ct.DISABLED)
        button2 = ct.CTkButton(self.model_frame, text="Invert Single", 
                               command=self.inv_single, width=50)
        button2.configure(state=ct.DISABLED)
        button3 = ct.CTkButton(self.model_frame, text="Invert All", 
                               command=self.open_map_window, width=50)
        button3.configure(state=ct.DISABLED)
        button4 = ct.CTkButton(self.model_frame, text="Plot Results", 
                               command=self.plot_1D_model, width=50)
        button4.configure(state=ct.DISABLED)
        self.distribute_buttons(self.model_frame, [button1, button2, button3, button4])
        self.model_frame_buttons = [button1, button2, button3, button4]

        # Button frame
        self.button_frame = ct.CTkFrame(self.main_frame, width=400)
        self.button_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ew")
        self.mode_button = ct.CTkButton(self.button_frame, text="Light Mode", command=self.toggle_mode)
        self.mode_button.grid(row=0, column=0, padx=5, pady=5)
    
    # Functions for defining app commands
    def open_da2(self):
        self.survey_index_entry.delete(0, ct.END)
        self.da2 = filedialog.askopenfilename(filetypes=[("Text files", "*.da2")])
        self.mo2 =  'my_model.mod'
        #os.path.splitext(os.path.basename(self.da2))[0] + '.mod'
        self.em2 = "my_model.emo" #os.path.splitext(os.path.basename(self.da2))[0] + '.em2'

        self.survey_list = S.read_da2_file(self.da2)

        self.n_soundings = len(self.survey_list)

        self.data_per_sounding = []

        for i in range(len(self.survey_list)):

            df = self.survey_list[i]['tem_data'].copy()

            df = df.drop(df.index[:3]).reset_index(drop=True)

            self.survey_list[i]['tem_data'] = df.copy()

            self.data_per_sounding.append(len(self.survey_list[i]['tem_data'].values))

        self.data_frame_buttons[2].configure(state=ct.NORMAL)
        self.data_frame_buttons[3].configure(state=ct.NORMAL)
        self.data_frame_buttons[4].configure(state=ct.NORMAL)
        self.survey_index_entry.configure(state=ct.NORMAL)
        self.survey_index_entry.insert(0, 0)

        self.layer_frame_buttons[0].configure(state=ct.NORMAL)
        self.model_type_combo.configure(state=ct.NORMAL)

    def open_map_window(self):

        # Create a new top-level window for the map using CustomTkinter
        map_window = ct.CTkToplevel(self.root)
        map_window.title("Map Window")
        #map_window.geometry("800x600")  # Optional: Set the size of the window

        #map_window.focus_set()
        #map_window.lift()
        #map_window.attributes("-topmost", True)

        # Create a new figure for the map
        fig, ax = plt.subplots()

        S.plot_survey_locs(self.survey_list, ax=ax, basemap=self.connection, 
                           epsg='epsg:32629')
        
        fig.tight_layout()

        # Embed the plot into the CustomTkinter window
        canvas = FigureCanvasTkAgg(fig, master=map_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Add a close button using CustomTkinter
        button_frame = ct.CTkFrame(map_window)
        button_frame.pack(pady=10, fill="x")

        close_button = ct.CTkButton(button_frame, text="Close", command=map_window.destroy)
        close_button.pack(side="right", padx=5)

    def set_inv_dir(self):
        # Open a directory selection dialog
        self.inv_dir = filedialog.askdirectory(title="Select inv_dir")

        os.chdir(self.inv_dir)
        
        print(self.inv_dir)

    def plot_all(self, plot_dist=False):

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_yscale('log')

        signals = np.ones((len(self.survey_list), 
                           np.max(self.data_per_sounding))).T * np.nan

        self.x = []
        self.y = []

        for i in range(self.n_soundings):
            self.x.append(self.survey_list[i]['header_info']['XUTM'])
            self.y.append(self.survey_list[i]['header_info']['YUTM'])

            signal = self.survey_list[i]['tem_data'].iloc[:,1].values

            signals[:len(signal),i] = signal

        dist = np.cumsum((np.diff(self.x) ** 2 + np.diff(self.y) ** 2) ** 0.5)
        dist = np.insert(0, 0, dist)

        if plot_dist:
            for i in range(np.max(self.data_per_sounding)):
                ax.plot(dist, np.abs(signals[i,:]), marker='x')
                ax.set_xlabel('Distance [m]')
                ax.set_ylabel('dB/dt [V/m$^2$]')
        
        else:
            for i in range(np.max(self.data_per_sounding)):
                ax.plot(np.abs(signals[i,:]), marker='x')
                ax.set_xlabel('Sounding Index')
                ax.set_ylabel('dB/dt [V/m$^2$]')

        fig.tight_layout()

        # Clear previous widgets in the plot frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Set up canvas for plotting
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.canvas.draw()

    def plot_sounding(self):

        self.get_survey_index()

        self.df = self.survey_list[self.survey_index]['tem_data']

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('dB/dt [V/m$^2$]')

        ax.scatter(self.df['GateCenter'], self.df['Signal'], marker='x', c='C0')
        ax.scatter(self.df['GateCenter'], -self.df['Signal'], marker='x', c='C1')

        if self.fwd_response is not None:
            ax.plot(self.fwd_times, np.abs(self.fwd_response), c='k')
            ax.scatter(self.fwd_times, self.fwd_response, marker='o', c='C0')
            ax.scatter(self.fwd_times, -self.fwd_response, marker='o',  c='C1')
    
        fig.tight_layout()

        # Clear previous widgets in the plot frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Set up canvas for plotting
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.canvas.draw()

    def plot_1D_model(self):

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xscale('log')

        plot_sounding(self.profiler_list[0], ax=ax, doi=False)

        fig.tight_layout()

        # Clear previous widgets in the plot frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Set up canvas for plotting
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.canvas.draw()

    def create_start_model(self):
        self.layer_frame_buttons[1].configure(state=ct.NORMAL)
        self.layer_frame_buttons[2].configure(state=ct.NORMAL)

        self.model_frame_buttons[1].configure(state=ct.NORMAL)
        self.model_frame_buttons[2].configure(state=ct.NORMAL)

        self.get_model_type()
        self.n_layers = 2
        self.update_modify_layer_frame()

    def add_layer(self):
        self.get_model_values()
        self.model_params = self.model_values 
        self.n_layers += 1
        self.update_modify_layer_frame()

    def remove_layer(self):
        if self.n_layers > 1:
            self.n_layers -= 1
            self.update_modify_layer_frame()

    def get_model_type(self):
  
        self.model_type = self.model_type_combo.get()

        if self.model_type == 'Resistivity':
            self.param_labels = ['Layer', 'Thk [m]', 'Ω [Ohm.m]']
            self.default_params = ['10.0', '100.0']

        if self.model_type == 'Pelton':
            self.param_labels = ['Layer', 'Thk [m]', 'Ω [Ohm.m]', 'M', 'log(τ [s])', 'C']
            self.default_params = ['10.0', '100.0', '0.2', '-2.3', '0.2']

        if self.model_type == 'Double Pelton':
            self.param_labels = ['Layer', 'Thk [m]', 'Ω [Ohm.m]', 'M', 'τ [s]', 'C1', 'M2', 't2 [s]', 'C2']
            self.default_params = ['10.0', '100.0', '0.2', '0.3', '0.2', '0.2', '0.003', '0.2']

        if self.model_type == 'MPA':
            self.param_labels = ['Layer', 'Thk [m]', 'Ω [Ohm.m]', 'φ_max', 'T_φ', 'C']
            self.default_params = ['10.0', '100.0', '0.2', '0.3', '0.2']

        self.n_params = len(self.param_labels)-1

    def get_survey_index(self):
        try:
            self.survey_index = int(self.survey_index_entry.get())
        except ValueError:
            print("Please enter a valid integer.")

    def update_modify_layer_frame(self):
        # Clear the layer scrollable frame
        for widget in self.layer_scrollable_frame.winfo_children():
            widget.destroy()

        # Clear the previous layer entries data
        self.layer_entries = []

        for j in range(self.n_params+1):
            ct.CTkLabel(self.layer_scrollable_frame, text=self.param_labels[j]).grid(row=1, column=j)

        if self.model_params is None:
            self.model_params = np.tile(self.default_params, (self.n_layers, 1))

        if self.model_params.shape[0] < self.n_layers:
            self.model_params = np.vstack([self.model_params, self.default_params])

        if self.model_params.shape[1] != self.n_params:
            self.get_model_type()
            self.model_params = np.tile(self.default_params, (self.n_layers, 1))

        for i in range(self.n_layers):
            ct.CTkLabel(self.layer_scrollable_frame, text=f"{i+1}").grid(row=i+2, column=0)
             # Store each layer's entries in a list
            layer_entry_row = []
            for j in range(self.n_params):
                
                p_entry = ct.CTkEntry(self.layer_scrollable_frame, width=70)
                p_entry.grid(row=i+2, column=j+1)
                p_entry.insert(0, self.model_params[i, j])

                # Append each entry to the layer's list
                layer_entry_row.append(p_entry)
            
            self.layer_entries.append(layer_entry_row)

    def get_model_values(self):
        model_values = []
        
        for model_row in self.layer_entries:
            row_values = []
            for entry in model_row:
                try:
                    row_values.append(float(entry.get()))  # Convert each entry to float
                except ValueError:
                    row_values.append(None)  # Handle non-numeric input
            model_values.append(row_values)
        
        self.model_values = np.array(model_values)

        if self.model_type == 'Pelton':

            self.model_values[:,3] = 10**self.model_values[:,3]

    def run_fwd(self):

        print('Dummy function')

    def inversion_objective(self, params):
        
        print('Dummy function')

    def inv_single(self):
        self.get_survey_index()
        
        self.inv_method([self.survey_list[self.survey_index]]*2, single=True) # Wee bit naughty

    def inv_all(self):
        
        self.inv_method(self.survey_list, single=False)
        
    def inv_method(self, survey_list, single):

        S.write_tem_files(survey_list)

        if self.model_type == 'Resistivity':
            messagebox.showwarning("Error", "Please select the MPA model type.")
            self.model_type = 'MPA'
            self.model_type_combo.set('MPA')
            self.create_start_model()

        if self.inv_dir is None:
            self.set_inv_dir()

        self.get_model_values() 
      
        model = self.model_values

        start_model = {'num_layers': model.shape[0]}
        start_model['rho'] = model[:,1]
        start_model['phi_max'] = model[:,2]
        start_model['tau_peak'] = model[:,3]
        start_model['c'] = model[:,4]
        start_model['thk'] = model[:-1,0]
        start_model['depth'] = np.cumsum(start_model['thk']).tolist()
        start_model['rho_v'] = -1
        start_model['phi_max_v'] =-1
        start_model['tau_peak_v']=-1
        start_model['c_v']=-1

        start_model['alpha_r'] = 0.7
        start_model['d_r'] = 10
        start_model['p'] = 0.5
        start_model['depth_c'] = -1

        confile=r"SarayaConfile.con"

        ih.run_mpa_inv(S, inv_dir=self.inv_dir, confile=confile, mod_file=self.mo2,
                       start_model=start_model, survey_list=survey_list)
        
        models = S.read_emo_file(self.em2)

        self.profiler_list = []

        for i, model in enumerate(models):

            profiler = {}
            
            profiler['x'] = model['x']
            profiler['y'] = model['y']

            # Extract layer-specific parameters dynamically
            n_layers = self.n_layers
            profiler['rhos'] = model['parameters'].values[-1, 0:n_layers]
            profiler['phi_maxs'] = model['parameters'].values[-1, n_layers:2*n_layers]
            profiler['tau_peaks'] = model['parameters'].values[-1, 2*n_layers:3*n_layers]
            profiler['cs'] = model['parameters'].values[-1, 3*n_layers:4*n_layers]

            # Thickness of each layer, which has one less entry than the number of depths
            profiler['thk'] = model['parameters'].values[-1, 4*n_layers:]

            # Calculate bottom and top depths based on thickness
            profiler['bot_depths'] = np.cumsum(profiler['thk'])
            profiler['top_depths'] = np.insert(profiler['bot_depths'][:-1], 0, 0)

            # Other parameters
            profiler['doi'] = 100
            profiler['n_layers'] = len(profiler['top_depths'])
            self.profiler_list.append(profiler)

        if single:
            self.fwd_response = models[0]['fwd_response'].iloc[:,6].values
            self.fwd_times = models[0]['fwd_response'].iloc[:,0].values
            model = models[0]

        else:
            self.fwd_response = models[self.survey_index]['fwd_response'].iloc[:,6].values
            self.fwd_times = models[self.survey_index]['fwd_response'].iloc[:,0].values
            model = models[self.survey_index]

        self.plot_sounding()

        self.model_frame_buttons[3].configure(state=ct.NORMAL)

        model_array = model['parameters'].values[-1, :]

        model_array = np.append(model_array, 10.0)

        model_array = model_array.reshape(-1, self.n_layers).T
    
        # Move the last column to the front
        model_array = np.hstack((model_array[:, -1:], model_array[:, :-1]))

        self.model_params = model_array

        self.layer_entries = []
        for i in range(self.n_layers):
            
            ct.CTkLabel(self.layer_scrollable_frame, text=f"{i+1}").grid(row=i+2, column=0)
             # Store each layer's entries in a list
            layer_entry_row = []
            for j in range(self.n_params):
                
                p_entry = ct.CTkEntry(self.layer_scrollable_frame, width=70)
                p_entry.grid(row=i+2, column=j+1)
                p_entry.insert(0, self.model_params[i, j])

                # Append each entry to the layer's list
                layer_entry_row.append(p_entry)
            
            self.layer_entries.append(layer_entry_row)   

    def calc_misfit(self,  output=True):
        """
        Calculate the L2 norm of the misfit in log space with an offset to handle negative values.
        """
        d = self.df['Signal'].values
        e = self.df['Error'].values * self.df['Signal'].values
        f = self.walktem_response

        r = d - f

        # Calculate the L2 norm (Euclidean distance) between the log-transformed values
        chi2 = 10**np.mean(np.log10((r/e)**2)) / len(d)

        #if output:

        print(chi2)

        return chi2

    def toggle_mode(self):
        # Toggle between light and dark mode
        if ct.get_appearance_mode() == "Dark":
            ct.set_appearance_mode("Light")
            self.mode_button.configure(text="Dark Mode")
        else:
            ct.set_appearance_mode("Dark")
            self.mode_button.configure(text="Light Mode")

# Run the app
if __name__ == "__main__":
    app = PyTEMApp()



