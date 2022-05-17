'''
This program provides experimental data of suction test to be imported in
the inverse solver. The main operation implemented in this file are:
- importing raw data files
- reduce data

Author: @aflah elouneg --- 17/05/2022
version 1.0
'''

import csv
import numpy as np

def import_mean_file(FRAME_NUM, SOURCE_FOLDER):
    with open(f'{SOURCE_FOLDER}/TABLE/frame{FRAME_NUM}_result.csv', mode='r') as read_file:
        csv_reader = csv.reader(read_file, delimiter=',')
        data_source = list(csv_reader)
    return data_source


#==================================================
## Reduced data field
def export_data(FRAME_NUM, SOURCE_FOLDER):
    data_mean = np.array(import_mean_file(FRAME_NUM, SOURCE_FOLDER))
    data_mean_reduc = []
    data_index = [0, 1, 2, 5, 6, 9, 10] # data type to be picked up from table.

    for i, node in enumerate(data_mean[1:]):
        data = []
        for j, index in enumerate(data_index):
            data.append(float(data_mean[i+1, index]))
        data_mean_reduc.append(data)

    data_mean_reduc = np.array(data_mean_reduc)

    export_data = {
        'node_x' : data_mean_reduc[1:, 3], # correspending to index 5
        'node_y' : data_mean_reduc[1:, 4], # correspending to index 6
        'disp_x_mean' : data_mean_reduc[1:, 5], # correspending to index 9
        'disp_y_mean' : data_mean_reduc[1:, 6],  # correspending to index 10
    }

    return export_data
