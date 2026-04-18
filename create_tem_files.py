#%%
import sys 
import matplotlib.pyplot as plt

import re

import contextily as cx
import numpy as np
import os
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import itertools

import shutil
import stat

sys.path.append(r'C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\Pytem/')

import importlib

import survey as S

sys.path.append(
    r'C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\Pytem0')

from import_colors import getAarhusCols, getParulaCols
from plot_tem import *

importlib.reload(S)

%matplotlib qt

#%%Test

locs = ['Badioula', 'Bembou', 'Kalando', 'Saraya South', 'Saraya Northeast', 'Saraya Northwest', 'Majiera']

loc = locs[6]
if loc == 'Badioula':
    survey_list1 = S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Badioula_1102.da2')
    survey_list2 = S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Badioula_1202_1.da2')
    survey_list3 = S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Badioula_1202_2.da2')
    survey_list = survey_list1 + survey_list2 + survey_list3

if loc == 'Bembou':
    survey_list1 = S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Bembou_1502_1.da2')
    survey_list2 = S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Bembou_1502_2.da2')
    survey_list = survey_list1 + survey_list2

if loc == 'Kalando':
    survey_list = S.read_da2_file(r'C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Kalando_1402.da2')


if loc == 'Saraya South':
    survey_list1 = S.read_da2_file(r"c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Saraya_South_1302.da2")
    survey_list2 = S.read_da2_file(r"c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Saraya_South_1402.da2")
    survey_list = survey_list1 + survey_list2 

if loc == 'Saraya Northeast':
    survey_list = S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Saraya_Northeast_1702.da2')

if loc == 'Saraya Northwest':
    survey_list= S.read_da2_file(r'c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Saraya_Northwest_1302.da2')

if loc == 'Majiera':
    survey_list = S.read_da2_file(r"c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Saraya\da2\Majiera_1602.da2")

fig, axs = plt.subplots(1,2)

S.plot_survey_locs(survey_list, ax = axs[0], basemap=True, epsg='epsg:32629')
S.reorder_survey_points(survey_list=survey_list, start_index=0)
S.rename_surveys(survey_list, site_name=loc)
S.plot_survey_locs(survey_list, ax=axs[1], basemap=True, epsg='epsg:32629')

fig.tight_layout()

plt.savefig(f'{loc}_map.png')
#%%
inv_dir = r"C:\Users\pamcl\AarhusInv\inv_dir"
os.chdir(inv_dir)

S.clean_inv_dir(inv_dir, extension='.tem')
S.clean_inv_dir(inv_dir, extension='.emo')

S.write_tem_files(survey_list)
