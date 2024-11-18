"""
Created on Fri Apr  28 09:09:32 2023

@author: pm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import geopandas as gpd
import textwrap
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib.colors import BoundaryNorm, Normalize, LinearSegmentedColormap
from scipy.interpolate import interp1d


def load_borehole(borehole_path):
    """

    Parameters
    ----------
    borehole_path : TYPE
        DESCRIPTION.

    Returns
    -------
    borehole_dict : TYPE
        DESCRIPTION.

    """
    bh_df = pd.read_csv(borehole_path, sep='\t')

    bh_dict = {}

    bh_dict['id'] = bh_df['id'][0]
    bh_dict['n_layers'] = len(bh_df)
    bh_dict['x'] = bh_df['utm_x'].values[0]
    bh_dict['y'] = bh_df['utm_y'].values[0]
    bh_dict['elevation'] = bh_df['elev'].values[0]
    bh_dict['top_depths'] = bh_df['top_depths'].values
    bh_dict['bot_depths'] = bh_df['bot_depths'].values
    bh_dict['colors'] = bh_df['colors'].values
    bh_dict['descriptions'] = bh_df['lith_descriptions'].values

    return bh_dict

def load_borehole_excel(borehole_excel_path):
    excel_file = pd.ExcelFile(borehole_excel_path)
    sheet_names = excel_file.sheet_names
    
    bh_dicts = []
    
    for sheet_name in sheet_names:
        bh_dict = {}
        bh_df = excel_file.parse(sheet_name)
        bh_dict['id'] = bh_df['id'][0]
        bh_dict['n_layers'] = len(bh_df)
        bh_dict['x'] = bh_df['utm_x'].values[0]
        bh_dict['y'] = bh_df['utm_y'].values[0]
        if 'elev' in bh_dict:
            bh_dict['elevation'] = bh_df['elev'].values[0]
        else:
            bh_dict['elevation'] = 0
        bh_dict['top_depths'] = bh_df['top_depths'].values
        bh_dict['bot_depths'] = bh_df['bot_depths'].values
        bh_dict['colors'] = bh_df['colors'].values
        bh_dict['lith_names'] = bh_df['lith_names'].values
        bh_dicts.append(bh_dict)
        
    return bh_dicts

def load_waterstrike_excel(waterstrike_excel_path):
    excel_file = pd.ExcelFile(waterstrike_excel_path)
    sheet_names = excel_file.sheet_names

    ws_dicts = []

    for sheet_name in sheet_names:
        
       
        ws_df = excel_file.parse(sheet_name)
        
        for idx in ws_df.index:
            ws_dict = {}
            ws_dict['id'] = ws_df['id'][idx]
            ws_dict['x'] = ws_df['utm_x'].values[idx]
            ws_dict['y'] = ws_df['utm_y'].values[idx]
            ws_dict['elevation'] = ws_df['elev'].values[idx]
            ws_dict['depth'] = ws_df['water_strike'].values[idx]

            ws_dicts.append(ws_dict)
    return ws_dicts


def load_xyz(xyz_path, start_idx=0, end_idx=None, return_mod_df=False, row_idx=0):
    """

    Parameters
    ----------
    xyz_path : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    elevation : TYPE
        DESCRIPTION.
    rhos : TYPE
        DESCRIPTION.
    depths : TYPE
        DESCRIPTION.
    doi_con : TYPE
        DESCRIPTION.
    doi_standard : TYPE
        DESCRIPTION.

    """

    # Scan through file and find where model part of file starts
    f = open(xyz_path, "r")

    file_lines = []
    line = f.readline()

    while line != '':
        line = f.readline()
        file_lines.append(line)
        if 'LINE_NO' in line:
            row_idx = len(file_lines)

    # Read data into pandas data frame
    mod_df = pd.read_csv(xyz_path, delimiter=None, skiprows=row_idx)

    col_names = mod_df.columns[0].split()[1:]

    mod_df = mod_df[mod_df.columns[0]].str.split(expand=True)

    mod_df.columns = col_names

    if 'DATE' in col_names:
        mod_df = mod_df.drop(['DATE'], axis=1)

    if 'TIME' in col_names:
        mod_df = mod_df.drop(['TIME'], axis=1)

    mod_df = mod_df.astype(float)

    # Extract relevant columns from dataframe
    rho_cols = [col for col in mod_df.columns
                if 'RHO' in col and 'STD' not in col]

    depth_cols = [col for col in mod_df.columns
                  if 'DEP_BOT' in col and 'STD' not in col]

    if end_idx is None:
        mod_df = mod_df.iloc[start_idx:, :]
    else:
        mod_df = mod_df.iloc[start_idx:end_idx, :]

    rhos = mod_df[rho_cols].values

    depths = mod_df[depth_cols].values

    doi_con = mod_df['DOI_CONSERVATIVE'].values
    doi_standard = mod_df['DOI_STANDARD'].values

    x = mod_df['UTMX'].values
    y = mod_df['UTMY'].values
    elev = mod_df['ELEVATION'].values

    residual = mod_df['RESDATA'].values
    line_num = mod_df['LINE_NO'].astype(int).values
    
    if return_mod_df:
        return x, y, elev, rhos, depths, doi_con, doi_standard, residual, line_num, mod_df
    
    else:
        return x, y, elev, rhos, depths, doi_con, doi_standard, residual, line_num


def load_shp_file(shp_file_path):

    gdf = gpd.read_file(shp_file_path)
    

    profile_list = []

    for index, row in gdf.iterrows():
        geometry = row['geometry']

        # Check if the geometry is a LineString
        if geometry.geom_type == 'LineString':
            # Access the LineString coordinates
            profile_coords = list(geometry.coords)

            profile_list.append(np.array(profile_coords))

    return profile_list


def interpolate_points(x, y, distance=10):
    interpolated_points = []

    for i in range(len(x)-1):
        x1 = x[i]
        y1 = y[i]
        x2 = x[i+1]
        y2 = y[i+1]
        dx = x2 - x1
        dy = y2 - y1
        segments = int(np.sqrt(dx**2 + dy**2) / distance)

        if segments != 0:

            for j in range(segments + 1):
                xi = x1 + dx * (j / segments)
                yi = y1 + dy * (j / segments)
                interpolated_points.append((xi, yi))

    interpolated_points = np.array(interpolated_points)

    xi, yi = interpolated_points[:, 0], interpolated_points[:, 1]
    dists = (np.diff(xi) ** 2 + np.diff(yi) ** 2) ** 0.5
    dists = np.cumsum(np.insert(dists, 0, 0))
    
    m = np.unique(dists, return_index=True)[1]
  

    return xi[m], yi[m], dists[m]


def interp_idw(x, y, z, xi, yi, power=2, interp_radius=10):
    """

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    xi : TYPE
        DESCRIPTION.
    yi : TYPE
        DESCRIPTION.
    power : TYPE, optional
        DESCRIPTION. The default is 2.
    interp_radius : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    """
    # Calculate distances between grid points and input points
    dists = np.sqrt((x[:, np.newaxis] - xi[np.newaxis, :])**2 +
                    (y[:, np.newaxis] - yi[np.newaxis, :])**2)

    # Calculate weights based on distances and power parameter
    weights = 1.0 / (dists + np.finfo(float).eps)**power

    # Set weights to 0 for points outside the specified radius
    weights[dists > interp_radius] = 0

    # Normalize weights for each grid point
    weights /= np.sum(weights, axis=0)

    # Interpolate values using weighted average
    zi = np.sum(z[:, np.newaxis] * weights, axis=0)
    return zi

def plot_sounding(walkTEM_dict, log=True, param='rhos', vmin=0, vmax=1000, title=None, label=None, ax=None, doi=True, col='k', ymin=None, ymax=None, extend_last_layer=True):
    
    params = walkTEM_dict[param]
    depths = walkTEM_dict['bot_depths']

    if extend_last_layer:
        params = np.append(params, params[-1])
        depths = np.append(depths, depths[-1]*2.5)
    
    if doi:
        doi = walkTEM_dict['doi']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,10))
    else:
        fig = ax.figure

    if doi:
        idx = np.where(depths > doi)[0][0]
        ax.step(params[:idx], -np.insert(depths, 0, 0)[:idx], where='pre', c=col, label=label)
        
        ax.step(params[idx-1:], -np.insert(depths, 0, 0)[idx-1:],
                where='pre', c='grey', ls='-',  alpha=0.8)
    else:
        ax.step(params, -np.insert(depths, 0, 0), where='pre', c=col, label=label)
    
    
    if log == True:
        ax.set_xscale('log')
        if vmin == 0:
            vmin = 1

    ax.set_xlim([vmin, vmax])
    ax.set_ylim([ymin, ymax])
    
    if label is not None:
        
        ax.legend()
        
    if title is not None:
        
        ax.set_title(title)

    if param == 'rhos':

        ax.set_xlabel('Resistivity [Ohm.m]')

    ax.set_ylabel('Elevation [m]')
    
    
    ax.grid(True, which='major')
    fig.tight_layout()
    


def plot_model2D(rhos, depths, elev=None, dists=None, doi=None, 
                 ax=None, vmin=1, vmax=1000, scale=10, contour=False,
                 cmap=plt.cm.viridis, n_bins=16, discrete_colors=False,
                 log=True, flip=False, cbar=True, cbar_orientation='vertical',
                 zmin=None, zmax=None, xmin=None, xmax=None, plot_title='',
                 cbar_label='Resistivity [Ohm.m]'):
    """

    Parameters
    ----------
    rhos : TYPE
        DESCRIPTION.
    depths : TYPE
        DESCRIPTION.
    elev : TYPE
        DESCRIPTION.
    dists : TYPE
        DESCRIPTION.
    doi : TYPE
        DESCRIPTION.
    vmin : TYPE, optional
        DESCRIPTION. The default is 1.
    vmax : TYPE, optional
        DESCRIPTION. The default is 200.
    hx : TYPE, optional
        DESCRIPTION. The default is 3.5.
    scale : TYPE, optional
        DESCRIPTION. The default is 10.
    cmap : TYPE, optional
        DESCRIPTION. The default is plt.cm.viridis.
    n_bins : TYPE, optional
        DESCRIPTION. The default is 11.
    log : TYPE, optional
        DESCRIPTION. The default is True.
    flip : TYPE, optional
        DESCRIPTION. The default is False.
    zmin : TYPE, optional
        DESCRIPTION. The default is None.
    zmax : TYPE, optional
        DESCRIPTION. The default is None.
    xmin : TYPE, optional
        DESCRIPTION. The default is None.
    xmax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    # Add extra distance, otherwise problem later on, check why this is...
    if dists is not None:
        dists = np.append(dists, dists[-1])
        plot_model_idx = False
    else:
        plot_model_idx = True

    # Add 0 m depth to depths, could be buggy
    depths = -np.c_[np.zeros(depths.shape[0]), depths]

    n_layers = rhos.shape[1]
    n_models = rhos.shape[0]

    # Transform data and lims into log
    if log:
        rhos = np.log10(rhos)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

    if plot_model_idx:
        x = np.arange(0, rhos.shape[0]+1)
        elev = np.zeros_like(x)[:-1]

    else:
        x = dists

    if flip:
        x = x[-1] - x[::-1]
        elev = elev[::-1]
        rhos = rhos[::-1, :]
        doi = doi[::-1]

    # Create boundary of polygons to be drawn
    xs = np.tile(np.repeat(x, 2)[1:-1][:, None], n_layers+1)

    depths = np.c_[np.zeros(depths.shape[0]), depths]
    ys = np.repeat(depths, 2, axis=0) + np.repeat(elev, 2, axis=0)[:, None]
    verts = np.c_[xs.flatten('F'), ys.flatten('F')]

    n_vert_row = verts.shape[0]
    connection = np.c_[np.arange(n_vert_row).reshape(-1, 2),
                       2*(n_models) +
                       np.arange(n_vert_row).reshape(-1, 2)[:, ::-1]]

    ie = (connection >= len(verts)).any(1)
    connection = connection[~ie, :]
    coordinates = verts[connection]

    # Determine color of each polygon
    cmaplist = [cmap(i) for i in range(cmap.N)]

    if discrete_colors:
        cmap = LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        bounds = np.linspace(vmin, vmax, n_bins)
        norm = BoundaryNorm(bounds, cmap.N)

    else:
        cmap = LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, 256)

        # define the bins and normalize
        bounds = np.linspace(vmin, vmax, 256)
        norm = BoundaryNorm(bounds, 256)

    # Create polygon collection
    coll = PolyCollection(coordinates, array=rhos.flatten('F'),
                          cmap=cmap, norm=norm, edgecolors=None)
    
    coll.set_clim(vmin=vmin, vmax=vmax)

    # Add polygons to plot
    if ax is None:
        hx = 3.5
        fig, ax = plt.subplots(1, 1, figsize=(hx/(100/dists[-1] * scale),
                                              hx*1.1))
    else:
        fig = ax.figure

    if contour:
        max_depth = 100
        centroid = np.mean(coordinates, axis=1)
        centroidx = centroid[:, 0].reshape((-1, n_models))
        centroidz = centroid[:, 1].reshape((-1, n_models))
        xc = np.vstack([centroidx[0, :], centroidx, centroidx[-1, :]])
        zc = np.vstack([np.zeros(n_models), centroidz, -
                       np.ones(n_models)*max_depth])
        val = np.c_[rhos[:, 0], rhos, rhos[:, -1]].T

        levels = np.linspace(vmin, vmax, 15)

        ax.contourf(xc, zc, val, cmap=cmap, levels=levels, extend='both')

    else:
        ax.add_collection(coll)

    # Blank out models below doi
    if doi is not None:
        doi = (np.repeat(elev, 2) - np.repeat(doi, 2)).tolist()

        doi.append(-1000)
        doi.append(-1000)
        doi.append(doi[-1])

        x_doi = xs[:, 0].tolist()

        x_doi.append(x_doi[-1])
        x_doi.append(x_doi[0])
        x_doi.append(x_doi[0])

        ax.fill(np.array(x_doi),  np.array(doi), edgecolor="none",
                facecolor='w', alpha=0.9)

    if dists is not None:
        ax.set_aspect(scale)
        ax.set_xlabel('Distance [m]\n')
    else:
        ax.set_xlabel('Index')

    ax.set_ylabel('Elevation [m]')

    if cbar:
        
        if cbar_orientation == 'vertical':
            cbar = fig.colorbar(coll, label=cbar_label, ax=ax,
                                orientation=cbar_orientation, shrink=0.8)
            
        else:
            cbar = fig.colorbar(coll, label=cbar_label, ax=ax,
                                orientation=cbar_orientation, shrink=0.7, fraction=0.06, pad=0.2)

        if log:
            tick_locs = np.arange(int(np.floor(vmin)), int(np.ceil(vmax)))

            if tick_locs[-1] < vmax:
                tick_locs = np.append(tick_locs, vmax)
                
            if tick_locs[0] < vmin:
                tick_locs = np.append(vmin, tick_locs[1:])

            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(np.round(10**tick_locs).astype(int))
            cbar.ax.minorticks_off()

        else:
            tick_locs = np.arange(vmin, vmax+0.00001, int(vmax-vmin)/4)
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(np.round(tick_locs))
            cbar.ax.minorticks_off()
            #cbar.ax.tick_params(size=0)

    if zmin is None:
        zmin = np.nanmin(elev)+np.min(depths)

    if zmax is None:
        zmax = np.nanmax(elev)

    if xmin is None:
        xmin = 0

    if xmax is None:
        if dists is not None:
            xmax = dists[-1]
        else:
            xmax = n_models

    ax.set_ylim([zmin, zmax])
    ax.set_xlim([xmin, xmax])

    if len(plot_title) != 0:
        ax.set_title(plot_title)

    ax.grid(which='both')
    fig.tight_layout()


def interp_lin(x, z):
    """


    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    zi : TYPE
        DESCRIPTION.

    """

    # Create interpolation function, excluding nan values
    interp_func = interp1d(x[~np.isnan(z)], z[~np.isnan(z)],
                           kind='linear', fill_value="extrapolate")
    zi = z.copy()
    zi[np.isnan(z)] = interp_func(x[np.isnan(z)])
    
    if np.isnan(zi[-1]):
        zi[-1] = zi[-2]

    return zi


def find_nearest(dict_data, x, y, dists=None, elev=None):
    """

    Parameters
    ----------
    dict_data : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    dists : TYPE
        DESCRIPTION.
    elev : TYPE
        DESCRIPTION.

    Returns
    -------
    dist_loc : TYPE
        DESCRIPTION.
    elev_loc : TYPE
        DESCRIPTION.
    min_dist : TYPE
        DESCRIPTION.

    """

    x_loc = dict_data['x']
    y_loc = dict_data['y']

    idx = np.argmin(((x_loc - x) ** 2 + (y_loc - y) ** 2) ** 0.5)
    min_dist = np.min(((x_loc - x) ** 2 + (y_loc - y) ** 2) ** 0.5)

    if elev is not None:
        elev_loc = elev[idx]
    else:
        elev_loc = dict_data['elevation']
    
    if dists is not None:
        
        dist_loc = dists[idx]
        
    else:
        dist_loc = np.nan

    return dist_loc, elev_loc, min_dist, idx

def add_borehole(bh_dict, ax, bh_width=0.2, text_size=12, x_start=None, x_end=None, bounds=False):
    """
    

    Parameters
    ----------
    bh_list : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    bh_width : TYPE, optional
        DESCRIPTION. The default is dists[-1]/100.

    Returns
    -------
    None.

    """
    
    if x_start is None:
        x_start = ax.get_xlim()[0]
        x_end = 10**(np.log10(ax.get_xlim()[1]) * bh_width)

    elev = bh_dict['elevation']

    print(elev)
            
    for i in range(bh_dict['n_layers']):
                
            coordinates = np.array(([x_start, elev-bh_dict['top_depths'][i]],
                                    [x_end, elev-bh_dict['top_depths'][i]],
                                    [x_end, elev-bh_dict['bot_depths'][i]],
                                    [x_start, elev-bh_dict['bot_depths'][i]],
                                    [x_start, elev-bh_dict['top_depths'][i]]))
            
            if bounds:

                p = Polygon(coordinates, facecolor=bh_dict['colors'][i], edgecolor = 'k', lw=0.5)

            else:
                p = Polygon(coordinates, facecolor=bh_dict['colors'][i], edgecolor = 'k', lw=0)
        
            ax.add_patch(p)
            
    coordinates = np.array(([x_start, elev-bh_dict['top_depths'][0]],
                            [x_end, elev-bh_dict['top_depths'][0]],
                            [x_end, elev-bh_dict['bot_depths'][-1]],
                            [x_start, elev-bh_dict['bot_depths'][-1]],
                            [x_start, elev-bh_dict['top_depths'][-1]]))
            
    p = Polygon(coordinates, facecolor='none', edgecolor = 'k', lw=1)
    
    ax.add_patch(p)
    
    
def write_boreholes(bh_list):
    
    for bh in bh_list:

        bh_array = np.ones((bh['n_layers'], 9))
        
        bh_df = pd.DataFrame(bh_array)
        
        bh_df.iloc[:,0] = bh['id']
        bh_df.iloc[:,1] = bh['x']
        bh_df.iloc[:,2] = bh['y']
        bh_df.iloc[:,3] = bh['elevation']
        bh_df.iloc[:,4] = bh['top_depths']
        bh_df.iloc[:,5] = bh['bot_depths']
        bh_df.iloc[:,6] = bh['simple']
        bh_df.iloc[:,7] = bh['colors']
        bh_df.iloc[:,8] = bh['lith_names']
        
        bh_df.columns = ['id', 'utm_x', 'utm_y', 'elev', 'top_depths',
                           'bot_depths', 'lith_names', 'colors', 'lith_descriptions']
        
        bh_df.to_csv(bh['id']+'.dat', index=False, sep='\t')
        
        

def add_boreholes(bh_list, x, y, dists, ax,  elev=None, search_radius=150,
                  bh_width=50, add_label=False, text_size=12, shift=5, bounds=False):
    """

    Parameters
    ----------
    bh_list : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    bh_width : TYPE, optional
        DESCRIPTION. The default is dists[-1]/100.

    Returns
    -------
    None.

    """

    for bh in bh_list:

        dist_loc, elev_loc, min_dist, idx = find_nearest(bh, x, y, dists, elev)


        if min_dist < search_radius:
            x1 = dist_loc - bh_width/2
            x2 = dist_loc + bh_width/2

            for i in range(bh['n_layers']):
                verts = np.array(([x1, elev_loc - bh['top_depths'][i]],
                                  [x2, elev_loc - bh['top_depths'][i]],
                                  [x2, elev_loc - bh['bot_depths'][i]],
                                  [x1, elev_loc - bh['bot_depths'][i]],
                                  [x1, elev_loc - bh['top_depths'][i]]))

                if bounds:
                    p = Polygon(verts, facecolor=bh['colors'][i], edgecolor = 'k', lw=0.5)

                else:
                    p = Polygon(verts, facecolor=bh['colors'][i], edgecolor = 'k', lw=0)
                
                ax.add_patch(p)

            # add boundary around log
            verts = np.array(([x1, elev_loc - bh['top_depths'][0]],
                              [x2, elev_loc - bh['top_depths'][0]],
                              [x2, elev_loc - bh['bot_depths'][-1]],
                              [x1, elev_loc - bh['bot_depths'][-1]],
                              [x1, elev_loc - bh['top_depths'][0]]))

            p = Polygon(verts, facecolor='none', edgecolor='k', lw=1)

            ax.add_patch(p)
            if add_label:
                ax.text(dist_loc, elev_loc+shift,  bh['id'][4:],
                        horizontalalignment='center',
                        verticalalignment='top', fontsize=text_size)

        else:
            print(bh['id'] + ' is ' + str(np.round(min_dist/1000, 2)) +
                  ' km from profile, it was not included.')
            
            
def add_waterstrikes(ws_list, x, y, dists, ax, elev = None, search_radius=50, col='blue',
                     ws_width=50, add_label=False, text_size=12):
    """

    Parameters
    ----------
    ws_list : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    bh_width : TYPE, optional
        DESCRIPTION. The default is dists[-1]/100.

    Returns
    -------
    None.

    """

    for ws in ws_list:

        dist_loc, elev_loc, min_dist, idx = find_nearest(ws, x, y, dists, elev)

        if min_dist < search_radius:
            x1 = dist_loc - ws_width/2
            x2 = dist_loc + ws_width/2

            ax.plot((x1, x2), (elev_loc - ws['depth'], elev_loc - ws['depth']), 
                    lw=1.5, c=col, ls=(0, (1, 1)))

        else:
            print(ws['id'] + ' is ' + str(np.round(min_dist/1000, 2)) +
                  ' km from profile, it was not included.')


def get_colors(rhos, vmin, vmax, cmap=plt.cm.viridis, n_bins=16, log=True,
               discrete_colors=False):
    """
    Return colors from a color scale based on numerical values

    Parameters
    ----------
    rhos : TYPE
        DESCRIPTION.
    vmin : TYPE
        DESCRIPTION.
    vmax : TYPE
        DESCRIPTION.
    cmap : TYPE, optional
        DESCRIPTION. The default is 'viridis'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if log:
        rhos = np.log10(rhos)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

    # Determine color of each polygon
    cmaplist = [cmap(i) for i in range(cmap.N)]

    if discrete_colors:
        cmap = LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        bounds = np.linspace(vmin, vmax, n_bins)
        norm = BoundaryNorm(bounds, cmap.N)

    else:
        cmap = LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, 256)

        # define the bins and normalize
        bounds = np.linspace(vmin, vmax, 256)
        norm = BoundaryNorm(bounds, 256)

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    return scalar_map.to_rgba(rhos)


def add_walkTEM(walkTEM_list, x, y, dists, vmin, vmax, ax=None, elev=None, add_cbar=False,
                log=True, cmap='viridis', n_bins=16, discrete_colors=False, cbar_orientation='vertical',
                search_radius=100, walkTEM_width=100):
    """

    Parameters
    ----------
    bh_list : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    bh_width : TYPE, optional
        DESCRIPTION. The default is dists[-1]/100.

    Returns
    -------
    None.

    """
    
    if ax is None:
        hx = 3.5
        fig, ax = plt.subplots(1, 1, figsize=(hx/(100/dists[-1] * scale),
                                              hx*1.1))
    else:
        fig = ax.figure

    for walkTEM in walkTEM_list:

        dist_loc, elev_loc, min_dist, idx = find_nearest(walkTEM, x, y,
                                                         dists, elev)

        if min_dist < search_radius:
            x1 = dist_loc - walkTEM_width/2
            x2 = dist_loc + walkTEM_width/2

            for i in range(walkTEM['n_layers']):
                verts = np.array(([x1, elev_loc - walkTEM['top_depths'][i]],
                                  [x2, elev_loc - walkTEM['top_depths'][i]],
                                  [x2, elev_loc - walkTEM['bot_depths'][i]],
                                  [x1, elev_loc - walkTEM['bot_depths'][i]],
                                  [x1, elev_loc - walkTEM['top_depths'][i]]))

                walkTEM['colors'] = get_colors(walkTEM['rhos'],
                                               vmin=vmin, vmax=vmax, log=log,
                                               cmap=cmap,
                                               discrete_colors=discrete_colors)

                if walkTEM['bot_depths'][i] > walkTEM['doi']:
                    p = Polygon(verts, facecolor=walkTEM['colors'][i],
                                alpha= 0.1, lw=0)

                else:
                    p = Polygon(verts, facecolor=walkTEM['colors'][i], lw=0)

                ax.add_patch(p)

            verts = np.array(([x1, elev_loc - walkTEM['top_depths'][0]],
                              [x2, elev_loc - walkTEM['top_depths'][0]],
                              [x2, elev_loc - walkTEM['bot_depths'][-1]],
                              [x1, elev_loc - walkTEM['bot_depths'][-1]],
                              [x1, elev_loc - walkTEM['top_depths'][0]]))

            p = Polygon(verts, facecolor='none', edgecolor='k', lw=0.5)

            ax.add_patch(p)
        
     

        else:
            print(walkTEM['name'] + ' is ' + str(np.round(min_dist/1000, 2)) +
                  ' km from profile, it was not included.')
    
    if log:
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
        
    
    if add_cbar:
        cmaplist = [cmap(i) for i in range(cmap.N)]
    
        
        
        cmap = LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, 256)
    
        # define the bins and normalize
     
        
        bounds = np.linspace(vmin, vmax, 256)
        norm = BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        if cbar_orientation == 'vertical':
            cbar = fig.colorbar(sm, label='Resistivity [Ohm.m]', ax=ax,
                                orientation=cbar_orientation, shrink=0.8)
            
        else:
            cbar = fig.colorbar(sm, label='Resistivity [Ohm.m]', ax=ax,
                                orientation=cbar_orientation, shrink=0.7, fraction=0.06, pad=0.2)
        
        tick_locs = np.arange(int(np.floor(vmin)), int(np.ceil(vmax)))
    
        if tick_locs[-1] < vmax:
            tick_locs = np.append(tick_locs, vmax)
            
        if tick_locs[0] < vmin:
            tick_locs = np.append(vmin, tick_locs[1:])
    
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(np.round(10**tick_locs).astype(int))
        cbar.ax.minorticks_off()


def load_walkTEM(walktem_path):
    """
    Parameters
    ----------
    walktem_path : TYPE
        DESCRIPTION.

    Returns
    -------
    walkTEM_list : TYPE
        DESCRIPTION.

    """
    x, y, elev, rhos, depths, doi_con, doi_standard, residual, line_num = load_xyz(walktem_path)

    n_models = rhos.shape[0]

    walkTEM_list = []

    for i in range(n_models):
        walkTEM = {}
        walkTEM['name'] = 'wt' + str(i+1)
        walkTEM['x'] = x[i]
        walkTEM['y'] = y[i]
        walkTEM['elevation'] = elev[i]
        walkTEM['bot_depths'] = depths[i]
        walkTEM['top_depths'] = np.insert(depths[i], 0, 0)
        walkTEM['rhos'] = rhos[i]
        walkTEM['n_layers'] = int(len(walkTEM['bot_depths']))
        walkTEM['residual'] = residual[i]
        walkTEM['doi'] = doi_standard[i]

        walkTEM_list.append(walkTEM)

    return walkTEM_list


def plot_key(bh_list, ax=None, max_line_length=50, title='Geological Key', label_names=None, drop_idx = None):
    
    lith_cols = []
    lith_names = []
    top_depths = []
    
    

    # Iterate through the list of dictionaries and extract unique values
    for bh in bh_list:
        lith_cols.append(bh['colors'].tolist())
        lith_names.append(bh['lith_names'].tolist())
        top_depths.append(bh['top_depths'].tolist())

    lith_cols = np.concatenate(lith_cols)
    lith_names = np.concatenate(lith_names)
    top_depths = np.concatenate(top_depths)

    
    unique_cols = np.unique(lith_cols)

    lith_key = []
    lith_depth = []
    for col in unique_cols:

        idx = np.where(lith_cols == col)[0]

        lith_key.append(lith_names[idx][0])
        lith_depth.append(np.mean(top_depths[idx]))

    idx = np.argsort(lith_depth)


    lith_key = np.array(lith_key)[idx]
    unique_cols = np.array(unique_cols)[idx]
    
    if label_names is not None:
        
        lith_key = label_names

    if ax is None:

        fig, ax = plt.subplots(1, 1)
        
    if drop_idx is not None:
        label_names.pop(drop_idx)
        unique_cols=unique_cols[drop_idx+1:]
        #print(lith_key)
        #lith_key.pop(drop_idx)
        
    # Iterate through the geological units and plot colored squares
    y_position = 0.5  # Initial y-position for the first square
    for i in range(len(unique_cols)):
        #print(i)
        # Plot a colored square
        ax.add_patch(plt.Rectangle((0, y_position), 0.4, 0.37, color=unique_cols[i]))
        #if len(lith_names) > 1:
        #    label = ' / '.join(lith_key[i])
            
        #else:
        label = lith_key[i]
        wrapped_label = '\n'.join(textwrap.wrap(label, max_line_length))

        ax.text(0.6, y_position + 0.15, wrapped_label, va='center', fontsize=12)

        # Update the y-position for the next square
        y_position -= 0.7

    # Set axis limits and labels
    ax.set_xlim(0, 4)
    ax.set_ylim(y_position, 1)
    ax.axis('off')  # Turn off axis
    ax.set_title(title, loc='left')

