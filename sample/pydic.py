"""
Created on Fri Apr 19 09:08:55 2019

@author: Quentin LUCOT

Updated on Jan 18 2021
by Audrey Bertin and Aflah Elouneg
"""

#########################################
###############Imports###################
#########################################
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
import sys

#########################################
#############Parameters##################
#########################################

def make_new_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise

def suction_dic(data_directory_path, pydic_methods, SAVE_VIDEO_FRAMES = False,
                    DISP_FIELD_IMAGES = False):
    '''
    data_directory_path : the folder containing videos for a single test series
    '''

    pydic = imp.load_source('pydic_methods', pydic_methods)
                # URL to import pydic_methods.py

    fps = 14. #video framerate (for Cutiscan : 14fps)
    image_dimensions = (1280,1024) # Video resolution (px)
    WS = 24
    GS = 20

    skin_diameter = 5.0 # ZOI diameter (mm)

    #Video ZOI (px)
    min_x = 154 #px
    max_x = 154 + 960 #px
    min_y = 62 #px
    max_y = 62 + 960 #px

    scale_disp = 5 # Displacement field vectors scale on images

    directories_path_list = []
    avi_files_path_list = []
    frames_directories_path_list = []
    tables_directories_path_list = []
    disps_directories_path_list = []

    #Video listing
    for root, dirs, files in os.walk(data_directory_path):
        for file in files:
            if file.endswith('avi') :
                    directories_path_list.append(root)
                    avi_files_path_list.append(os.path.join(root,file))
                    frames_folder = os.path.join(root, 'FRAME')
                    frames_directories_path_list.append(frames_folder)

    correl_wind_size = (WS, WS) # Size of the correlation windows (px)
    correl_grid_size = (GS, GS) # Size of the interval (dx,dy) of the correlation grid (px)

    #########################################
    ############Core Program#################
    #########################################
    #Center position of ZOI
    if (max_x+min_x)%2 !=0:
        max_x=max_x+1
    if (max_y+min_y)%2 !=0:
        max_y=max_y+1
    resolution=skin_diameter/(max_x-min_x)
    mean_x=(min_x+max_x)/2
    mean_y=(min_y+max_y)/2
    if (((max_x+min_x)/2-min_x)%correl_grid_size[0])!=0:
        if (min_x-correl_grid_size[0]+((max_x+min_x)/2-min_x)%correl_grid_size[0]) <0:
            min_x=min_x+((max_x+min_x)/2-min_x)%correl_grid_size[0]
        else:
            min_x=min_x-correl_grid_size[0]+((max_x+min_x)/2-min_x)%correl_grid_size[0]

    if ((max_x-min_x)%correl_grid_size[0])!=0:
        if (max_x+correl_grid_size[0]-(max_x-min_x)%correl_grid_size[0])>image_dimensions[0]:
            max_x=max_x-(max_x-min_x)%correl_grid_size[0]
        else:
            max_x=max_x+correl_grid_size[0]-(max_x-min_x)%correl_grid_size[0]

    if (((max_y+min_y)/2-min_y)%correl_grid_size[1])!=0:
        if (min_y-correl_grid_size[1]+((max_y+min_y)/2-min_y)%correl_grid_size[1]) <0:
            min_y=min_y+((max_y+min_y)/2-min_y)%correl_grid_size[1]
        else:
            min_y=min_y-correl_grid_size[1]+((max_y+min_y)/2-min_y)%correl_grid_size[1]

    if ((max_y-min_y)%correl_grid_size[1])!=0:
        if (max_y+correl_grid_size[1]-(max_y-min_y)%correl_grid_size[1])>image_dimensions[1]:
            max_y=max_y-(max_y-min_y)%correl_grid_size[1]
        else:
            max_y=max_y+correl_grid_size[1]-(max_y-min_y)%correl_grid_size[1]


    #Frame extraction
    for i in range(0, len(directories_path_list)):

        make_new_directory(frames_directories_path_list[i])
        cap = cv2.VideoCapture(avi_files_path_list[i])
        success,image = cap.read()
        count = 0
        print(f'Extracting new frames ...')
        while success:
            sys.stdout.write("=".format("="))
            sys.stdout.flush()
            # time.sleep(0.1)
            os.chdir(frames_directories_path_list[i])
            cv2.imwrite("frame%04d.png" % count, image)
            success,image = cap.read()
            count += 1
            # sys.stdout.write('\n')
        print(f'|{count} frames have been successfully extracted')
        cap.release()
        cv2.destroyAllWindows()

        #Pydic excetution and data scraping
        table_files_path_list=[]
        os.chdir(frames_directories_path_list[i])
        points = pydic.init("./*.png", correl_wind_size, correl_grid_size, "./result.dic",
                    area_of_intersest=((min_x, min_y),
                    (max_x+correl_grid_size[0], max_y+correl_grid_size[1])),)

        pydic.grid_list=[]
        pydic.read_dic_file('./result.dic',
                            interpolation='raw', save_image=DISP_FIELD_IMAGES,
                            scale_disp=scale_disp,
                            )
        for root, dirs, files in os.walk(frames_directories_path_list[i]):
            for file in files:
                if file.endswith('csv'):
                    table_files_path_list.append(os.path.join(root,file))
        data_angle_norm_disp=[]
        data_angle_disp_r=[]
        rval=[]
        for l in range (0,8):
            data_angle_norm_disp.append([])
            data_angle_disp_r.append([])
            rval.append(['frame number','time in seconds'])
        for j in range(0,len(table_files_path_list)):
            with open(table_files_path_list[j],mode='r') as read_file:
                csv_reader=csv.reader(read_file,delimiter=',')
                data=[]
                for l in range(0,8):
                    data_angle_norm_disp[l].append([j,round(float(j)/fps,3)])
                    data_angle_disp_r[l].append([j,round(float(j)/fps,3)])
                data=list(csv_reader)
                data[0]=['index','index_x','index_y','pos_x(px)','pos_y(px)','pos_x(mm)','pos_y(mm)','disp_x(px)', 'disp_y(px)','disp_x(um)','disp_y(um)','strain_xx','strain_yy','strain_xy','r(px)','r(mm)','theta','disp_r(px)','disp_theta(px)','disp_r(um)','disp_theta(um)','norm_disp(px)','norm_disp(um)']
                for k in range(1,len(data)):
                    data[k].append(0.)
                    data[k].append(float(data[k][7]))
                    data[k].append(-float(data[k][8]))
                    data[k].append(float(data[k][9]))
                    data[k][3]=float(data[k][3])-mean_x
                    data[k][4]=-float(data[k][4])+mean_y
                    data[k][7]=float(data[k][5])
                    data[k][8]=-float(data[k][6])
                    data[k][5]=resolution*data[k][3]
                    data[k][6]=resolution*data[k][4]
                    data[k][9]=float(data[k][7])*resolution*1000.
                    data[k][10]=float(data[k][8])*resolution*1000.
                    data[k].append(math.sqrt(data[k][3]**2+data[k][4]**2))
                    data[k].append(math.sqrt(data[k][5]**2+data[k][6]**2))
                    data[k].append(math.degrees(math.atan2(data[k][6],data[k][5])))
                    data[k].append(float(data[k][7])*math.cos(math.radians(float(data[k][16])))+float(data[k][8])*math.sin(math.radians(float(data[k][16]))))
                    data[k].append(-float(data[k][7])*math.sin(math.radians(float(data[k][16])))+float(data[k][8])*math.cos(math.radians(float(data[k][16]))))
                    data[k].append(float(data[k][17])*resolution*1000.)
                    data[k].append(float(data[k][18])*resolution*1000.)
                    data[k].append(math.sqrt(data[k][7]**2+data[k][8]**2))
                    data[k].append(float(data[k][21])*resolution*1000.)

                    for l in range (0,8):
                        angle=l*45-135
                        if data[k][16]==angle and float(data[k][15])<skin_diameter/2:
                            data_angle_norm_disp[l][j].append(data[k][22])
                            data_angle_disp_r[l][j].append(abs(float(data[k][19])))
                            if ("r = "+str(round(float(data[k][15]),3))+ "mm") not in rval[l]:
                                rval[l].append(("r = "+str(round(float(data[k][15]),3))+ "mm"))


            with open(table_files_path_list[j],mode='w',newline='') as write_file:
                csv_writer=csv.writer(write_file)
                csv_writer.writerows(data)

            read_file.close()
            write_file.close()

        tables_figures_path = os.path.join(frames_directories_path_list[i], 'disp')
        shutil.move(tables_figures_path, directories_path_list[i])

        disp_figures_path = os.path.join(frames_directories_path_list[i], 'result')
        shutil.move(disp_figures_path, directories_path_list[i])


        # os.remove(os.path.join(frames_directories_path_list[i], 'pydic'))
        os.remove(os.path.join(frames_directories_path_list[i], 'result.dic'))
        try:
            shutil.rmtree(os.path.join(directories_path_list[i], 'TABLE'),
                    ignore_errors=True)
        except:
            pass
        try:
            shutil.rmtree(os.path.join(directories_path_list[i], 'DISP_FIG'),
                    ignore_errors=True)
        except:
            pass

        os.rename(os.path.join(directories_path_list[i], 'result'),
                        os.path.join(directories_path_list[i], 'TABLE'))

        os.rename(os.path.join(directories_path_list[i], 'disp'),
                        os.path.join(directories_path_list[i], 'DISP_FIG'))

        if not SAVE_VIDEO_FRAMES:
            shutil.rmtree(frames_directories_path_list[i])
