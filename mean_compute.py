"""
Created on Feb 03, 2021 : 07:55
Updated on Oct 16, 2021 : 18:15

@author: Aflah ELouneg

This program compute the mean and standard devation values of
dispalcement fields gathred from digital image corralation of
ring suction test records.
"""

#########################################
### Imports

import csv
import os
import shutil
import errno
import zipfile
import cv2     #conda install cv2 or pip install cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import math
import imp
import time
import importlib
import copy

pydic = imp.load_source('pydic', './pydic.py') # URL to import pydic.py

#=====================================
### Parameters

MULTI_PRESSURE = False

frame = 41  # number of the frame corresponding to the quasi-static state
pressure_setpoint = 100 # mbar
scale_disp = 5 # Displacement field vectors scale on images
array_color = (0,0,255)
point_color = (255,255,0)

if MULTI_PRESSURE:
    pressure_setpoints = [[int(i),] for i in np.linspace(100, 500, 21)]
else:
    pressure_setpoints = [[pressure_setpoint,],]

file_csv = 'TABLE/frame00' + str(frame) + '_result.csv' # 4digits
reference_image = 'reference_image.png'

# ====================================
### Fixed parameters

image_dimensions = (1280,1024)
skin_diameter = 5.0 # ZOI diameter (mm)

WS = 72
GS = 20

# Video ZOI (px)
min_x = 150 #px
max_x = 1100 #px
min_y = 50 #px
max_y = 1000 #px

correl_grid_size = (GS, GS)


#==================================
# Setting directories

MAIN_DIR = os.getcwd()

try:
    os.makedirs('mean_values')
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir('mean_values'):
        pass
    else:
        raise

try:
    os.makedirs('mean_field_compute')
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir('mean_values'):
        pass
    else:
        raise

try:
    os.makedirs('std_values')
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir('mean_values'):
        pass
    else:
        raise

result_list=[]

print('Reading files ... :')

for i, p in enumerate(pressure_setpoints):
    result_list.append(p.copy())

    print("Searching for CSV files for p=", p[0])

    for root, dirs, files in os.walk(MAIN_DIR):
        for dir in dirs:
            if dir.startswith('p_' + str(p[0])):
                # dir = os.path.join(root, dir, 'p_' + str(p[0]))
                result_list[-1].append(os.path.join(root, dir, file_csv))
                # if i==0:
                #     reference_image = os.path.join(root, dir, reference_image_)


def prepare_saved_file(MAIN_DIR, prefix, pressure, extension):
    folder = os.path.join(MAIN_DIR, prefix)
    name = folder + '/p_' + str(pressure) + '.' + extension
    print("saving", name, "file...")
    return name


def draw_opencv(image, *args, **kwargs):
    """A function with a lot of named argument to draw opencv image
 - 'point' arg must be an array of (x,y) point
 - 'p_color' arg to choose the color of point in (r,g,b) format
 - 'pointf' to draw lines between point and pointf, pointf
   must be an array of same lenght than the point array
 - 'l_color' to choose the color of lines
 - 'grid' to display a grid, the grid must be a grid object
 - 'gr_color' to choose the grid color"""
    if type(image) == str :
         image = cv2.imread(image, 0)

    if 'text' in kwargs:
         text = kwargs['text']
         image = cv2.putText(image, text, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4)

    frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if  'point' in kwargs:
        p_color = (0, 255, 255) if not 'p_color' in kwargs else kwargs['p_color']
        for pt in kwargs['point']:
            if not np.isnan(pt[0]) and not np.isnan(pt[1]):
                 x = int(pt[0])
                 y = int(pt[1])
                 frame = cv2.circle(frame, (x, y), 4, p_color, -1)

    scale = 1. if not 'scale' in kwargs else kwargs['scale']
    if 'pointf' in kwargs and 'point' in kwargs:
        assert len(kwargs['point']) == len(kwargs['pointf']), 'bad size'
        l_color = (255, 120, 255) if not 'l_color' in kwargs else kwargs['l_color']
        for i, pt0 in enumerate(kwargs['point']):
            pt1 = kwargs['pointf'][i]
            if np.isnan(pt0[0])==False and np.isnan(pt0[1])==False and \
                    np.isnan(pt1[0])==False and np.isnan(pt1[1])==False :
                 disp_x = (pt1[0]-pt0[0])*scale
                 disp_y = (pt1[1]-pt0[1])*scale
                 frame = cv2.line(frame, (pt0[0], pt0[1]), (int(pt0[0]+disp_x),
                            int(pt0[1]+disp_y)), l_color, 2)

    if 'grid' in kwargs:
        gr =  kwargs['grid']
        gr_color = (255, 255, 255) if not 'gr_color' in kwargs else kwargs['gr_color']
        for i in range(gr.size_x):
            for j in range(gr.size_y):
                 if (not math.isnan(gr.grid_x[i,j]) and
                     not math.isnan(gr.grid_y[i,j]) and
                     not math.isnan(gr.disp_x[i,j]) and
                     not math.isnan(gr.disp_y[i,j])):
                      x = int(gr.grid_x[i,j]) + int(gr.disp_x[i,j]*scale)
                      y = int(gr.grid_y[i,j]) + int(gr.disp_y[i,j]*scale)

                      if i < (gr.size_x-1):
                           if (not math.isnan(gr.grid_x[i+1,j]) and
                               not math.isnan(gr.grid_y[i+1,j]) and
                               not math.isnan(gr.disp_x[i+1,j]) and
                               not math.isnan(gr.disp_y[i+1,j])):
                                x1 = int(gr.grid_x[i+1,j]) + int(gr.disp_x[i+1,j]*scale)
                                y1 = int(gr.grid_y[i+1,j]) + int(gr.disp_y[i+1,j]*scale)
                                frame = cv2.line(frame, (x, y), (x1, y1), gr_color, 2)

                      if j < (gr.size_y-1):
                           if (not math.isnan(gr.grid_x[i,j+1]) and
                               not math.isnan(gr.grid_y[i,j+1]) and
                               not math.isnan(gr.disp_x[i,j+1]) and
                               not math.isnan(gr.disp_y[i,j+1])):
                                x1 = int(gr.grid_x[i,j+1]) + int(gr.disp_x[i,j+1]*scale)
                                y1 = int(gr.grid_y[i,j+1]) + int(gr.disp_y[i,j+1]*scale)
                                frame = cv2.line(frame, (x, y), (x1, y1), gr_color, 4)
    if 'filename' in kwargs:
         cv2.imwrite( kwargs['filename'], frame)
         return

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', frame.shape[1], frame.shape[0])
    cv2.imshow('image',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ====================================
# Center position of ZOI

if (max_x + min_x)%2 !=0:
    max_x = max_x+1
if (max_y + min_y)%2 !=0:
    max_y = max_y+1
resolution = skin_diameter/(max_x - min_x)
mean_x = (min_x + max_x)/2
mean_y = (min_y + max_y)/2
if (((max_x + min_x)/2 - min_x)%correl_grid_size[0])!=0:
    if (min_x - correl_grid_size[0] + \
                ((max_x + min_x)/2 - min_x)%correl_grid_size[0]) <0:
        min_x = min_x + ((max_x + min_x)/2 - min_x)%correl_grid_size[0]
    else:
        min_x = min_x - correl_grid_size[0] +\
                ((max_x + min_x)/2 - min_x)%correl_grid_size[0]

if ((max_x - min_x)%correl_grid_size[0])!=0:
    if (max_x + correl_grid_size[0] - \
                (max_x - min_x)%correl_grid_size[0])>image_dimensions[0]:
        max_x = max_x - (max_x - min_x)%correl_grid_size[0]
    else:
        max_x = max_x + correl_grid_size[0] - \
                (max_x - min_x)%correl_grid_size[0]

if (((max_y + min_y)/2 - min_y)%correl_grid_size[1])!=0:
    if (min_y - correl_grid_size[1] + \
                ((max_y + min_y)/2 - min_y)%correl_grid_size[1]) <0:
        min_y = min_y + ((max_y + min_y)/2 - min_y)%correl_grid_size[1]
    else:
        min_y = min_y - correl_grid_size[1] + \
                ((max_y + min_y)/2 - min_y)%correl_grid_size[1]

if ((max_y - min_y)%correl_grid_size[1])!=0:
    if (max_y + correl_grid_size[1] - \
                (max_y - min_y)%correl_grid_size[1])>image_dimensions[1]:
        max_y = max_y - (max_y - min_y)%correl_grid_size[1]
    else:
        max_y = max_y + correl_grid_size[1] - (max_y-min_y)%correl_grid_size[1]


num_pressures = len(pressure_setpoints)
num_values = len(result_list[0]) - 1

for i, p in enumerate(pressure_setpoints):

    print(" Computing mean values for p=", p)

    with open(result_list[i][1], mode='r') as read_file:
        csv_reader = csv.reader(read_file, delimiter=',')
        data_source = list(csv_reader)

    num_variables = len(data_source[0])

    data_mean = copy.deepcopy(data_source)
    data_std = copy.deepcopy(data_source)
    num_nodes = len(data_source)

    for k in list(range(1, num_nodes)):
        for l in list(range(num_variables)):
            if l in [0, 1, 2, 3, 4]:
                data_mean[k][l] = int(float(data_source[k][l]))
                data_std[k][l] = int(float(data_source[k][l]))
            elif l in [5, 6]:
                data_mean[k][l] = float(data_source[k][l])
                data_std[k][l] = float(data_source[k][l])
            else:
                data_mean[k][l] = [float(data_source[k][l]),]
                data_std[k][l] = [float(data_source[k][l]),]

    for j in list(range(2, num_values + 1)):
        try:
            with open(result_list[i][j], mode='r') as read_file:
                csv_reader = csv.reader(read_file, delimiter=',')
                data_source = list(csv_reader)

                for k in list(range(1, num_nodes)):
                    for l in list(range(7, num_variables)):
                        data_mean[k][l].append(float(data_source[k][l]))
                        data_std[k][l].append(float(data_source[k][l]))
        except:
            print("Could not open the directory:", result_list[i][j])
            input('Press any key to continue ...')
            pass

    for k in list(range(1, num_nodes)):
        for l in range(7, num_variables):
            data_mean[k][l] = np.mean(data_mean[k][l])
            data_std[k][l] = np.std(data_std[k][l])

    with open(f'mean_values/p_{p[0]}.csv', mode='wt', newline='') as write_file:
        csv_writer=csv.writer(write_file)
        csv_writer.writerows(data_mean)
        read_file.close()
        write_file.close()

    with open(f'std_values/p_{p[0]}.csv', mode='wt', newline='') as write_file:
        csv_writer=csv.writer(write_file)
        csv_writer.writerows(data_std)
        read_file.close()
        write_file.close()


    name = prepare_saved_file(MAIN_DIR, 'mean_field_compute', p, 'png')
    reference_point = []
    correlated_point = []
    for data_point in data_mean[1:]:

        reference_point.append(np.array([np.float32(data_point[3] + mean_x),
                                np.float32(-data_point[4] + mean_y)]))
        correlated_point.append(np.array([np.float32(data_point[7] + data_point[3] + mean_x),
                                np.float32(-data_point[8] - data_point[4] + mean_y)]))

    draw_opencv(reference_image, point=reference_point,
                    pointf=correlated_point, l_color=array_color, p_color=point_color,
                    scale=scale_disp, filename=name, text=name)
