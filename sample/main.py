'''
The purpose of this program is to identify skin material parameters from
multiaxial ring suction test data. The main operations are:
- Data preparation: import cartestian grids of experimental data
    in quasi-static state
- Identification of parameters using a LSQ (leasts squares) method:
    Newton.
- Plot model fit curves (model vs. data)

=============================
Author: Aflah Elouneg :
            --  aflah.elouneg@uni.lu
            --  aflah.elouneg@femto-st.fr
Version 1.0:    17/05/2022
=============================
Version 1.1:    15/03/2024
- Add color maps of radial displacement field
- Adapt input/output foldes
- Remove unnecessary libraries
- Add enable/disable features
- Correct the displacement arrays in the function "ellipse_deformed"
- Add waiting message for long process

=============================
version 1.2:    08/06/2024
- Add the choice of DIC PROCESS (True/False) from terminal
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import errno
import newton
import itertools
import threading
import sys
import time
import matplotlib
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.pyplot import ticklabel_format
from matplotlib.ticker import MaxNLocator

np.set_printoptions(precision=6)
matplotlib.use('agg')

# ============================================
### Parameters

PLOT_DISP_FIELD = True
PLOT_DEFORMATION = True
PLOT_MODEL_FIT = True
IDENTIFY_PARAMETERS = True
SAVE_IDENTIFIED_PARAMETERS = True
FIGURES_FOLDER = 'output'
SOURCE_FOLDER = 'input'
SFF = 'svg' # saved file format : svg, eps, png, ...
FRAME_NUM = '0041' # frame number correspending to quasi-static state
pressure = 300 #mbar : suction pressure fixed for the test

# ================================================
### Functions

def make_new_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise


def interpolate_r_th(px, py, nodes_x, nodes_y, data_x, data_y):

    disp_interp = []
    XG = nodes_x
    YG = nodes_y
    disp = []

    for ux, uy in zip(data_x, data_y):
        disp.append([ux, uy])
    for X, Y in zip(px, py):
        disp_interp.append(griddata(list(zip(XG, YG)),
                                    disp, (X, Y), method='linear'))

    disp_interp = np.array(disp_interp)
    return disp_interp[:, 0], disp_interp[:, 1]


def interpolate_heatmap(px, py, nodes_x, nodes_y, data_x, data_y):

    disp_interp = []
    XG = nodes_x
    YG = nodes_y
    disp = []

    for ux, uy in zip(data_x, data_y):
        disp.append([ux, uy])
    for X, Y in zip(px, py):
        disp_interp.append(griddata(list(zip(XG, YG)),
                                    disp, (X, Y), method='linear'))

    disp_interp = np.array(disp_interp)
    return disp_interp[:, :, 0], disp_interp[:, :, 1]


def ellipse_deformed(X, Y, U, V):
    '''return final configuration of X, Y nodes after displacement U, V '''

    ellip_x = []
    ellip_y = []

    for x, y, u, v in zip(X, Y, U, V):
        phi = np.arctan2(y, x)
        n = np.sqrt(x**2 + y**2)
        ur = u*n*np.cos(phi)/n + v*n*np.sin(phi)/n
        urx = ur*np.cos(phi)
        ury = ur*np.sin(phi)

        ellip_x.append(x + urx*1e-3)
        ellip_y.append(y + ury*1e-3)

    return np.array(ellip_x), np.array(ellip_y)


def plot_axis_rotation(params, exp_nodes, r, angles):

    angles = np.append(angles, angles[0])
    params = np.stack(params, axis=1)[0]

    a   = params[0]
    b   = params[1]
    t0  = params[2]
    x0  = params[3]
    y0  = params[4]

    t0 *= np.pi/180.0

    exp_nodes_x = exp_nodes[0]
    exp_nodes_y = exp_nodes[1]

    model_nodes_x = a*np.cos(angles - t0)
    model_nodes_y = b*np.sin(angles - t0)

    model_nodes_x_rot = model_nodes_x*np.cos(t0) - model_nodes_y*np.sin(t0)
    model_nodes_y_rot = model_nodes_x*np.sin(t0) + model_nodes_y*np.cos(t0)

    x_undeform = r*np.cos(angles)
    y_undeform = r*np.sin(angles)

    data = [exp_nodes_x, exp_nodes_y]
    data[0] = np.append(data[0], data[0][0])
    data[1] = np.append(data[1], data[1][0])
    model = [model_nodes_x_rot, model_nodes_y_rot]

    # =======================================
    fig_name = f'Experimental VS. model fit (ellipses)'
    fh = plt.figure(fig_name)
    fh.clear()
    ax = fh.add_subplot(111)

    # ======================
    ax.plot(x_undeform, y_undeform, 'k-', linewidth=1.0, label='undeformed')
    ax.plot(data[0], data[1], 'r-', linewidth=1.0, label='data (deformed)')

    # ======================
    # plot arrows
    ax.arrow(x0, y0, x0 + a*np.cos(t0), y0 + a*np.sin(t0), color='r',
        linestyle='-', length_includes_head=True, width=0.005, head_width=0.025)
    ax.text(x0 + r*np.cos(t0) - 0.15, y0 + r*np.sin(t0), r'$\mathbf{e}_1$')
    ax.text(x0 + r*np.cos(t0)*0.5 - 0.05, y0 + r*np.sin(t0)*0.5 + 0.05, 'a')
    ax.arrow(x0, y0, x0 + b*np.cos(t0 + 0.5*np.pi),
        y0 + b*np.sin(t0 + 0.5*np.pi), color='r', linestyle='-',
        length_includes_head=True, width=0.005, head_width=0.025)
    ax.text(x0 + r*np.cos(t0 + 0.5*np.pi) - 0.05,
        y0 + r*np.sin(t0 + 0.5*np.pi) - 0.1, r'$\mathbf{e}_2$')
    ax.text(x0 + r*np.cos(t0 + 0.5*np.pi)*0.5 - 0.1,
        y0 + r*np.sin(t0 + 0.5*np.pi)*0.5 - 0.05, 'b')
    ax.arrow(x0, y0, x0 + r*np.cos(0.0), y0 + r*np.sin(0.0), color='k',
        linestyle='-', length_includes_head=True, width=0.005, head_width=0.025)
    ax.text(x0 + r*np.cos(0.0) - 0.2, y0 + r*np.sin(0.0) + 0.05,
        r'$\mathbf{e}^{\prime}_1$')
    ax.arrow(x0, y0, x0 + r*np.cos(0.5*np.pi), y0 + r*np.sin(0.5*np.pi),
                color='k', linestyle='-', length_includes_head=True,
                width=0.005, head_width=0.025)
    ax.text(x0 + r*np.cos(0.5*np.pi) - 0.15, y0 + r*np.sin(0.5*np.pi) - 0.1,
                r'$\mathbf{e}^{\prime}_2$')

    # plot angle corner
    points = np.linspace(0, t0, 50)
    r_ = r*0.1

    ax.plot(r_*np.cos(points), r_*np.sin(points), 'b')
    mid_point = 0.5*t0
    ax.text(1.25*r_*np.cos(mid_point),
        1.25*r_*np.sin(mid_point)-0.02, r'$\phi$')

    ax.set_xlabel(r"$x_1'$ [mm]", fontsize=12)
    ax.set_ylabel(r"$x_2'$ [mm]", fontsize=12)

    ax.legend(framealpha=0.2, bbox_to_anchor=[0.5, 0.25], loc='center')
    plt.gca().set_aspect("equal")

    plt.savefig(f'{FIGURES_FOLDER}/axis_rotation.{SFF}', bbox_inches='tight')


def plot_model_fit_ellipse(params, exp_nodes, r, angles):

    angles = np.append(angles, angles[0])
    params = np.stack(params, axis=1)[0]

    a   = params[0]
    b   = params[1]
    t0  = params[2]
    x0  = params[3]
    y0  = params[4]

    t0 *= np.pi/180.0

    exp_nodes_x = exp_nodes[0]
    exp_nodes_y = exp_nodes[1]

    model_nodes_x = a*np.cos(angles - t0)
    model_nodes_y = b*np.sin(angles - t0)

    model_nodes_x_rot = model_nodes_x*np.cos(t0) - model_nodes_y*np.sin(t0) - x0
    model_nodes_y_rot = model_nodes_x*np.sin(t0) + model_nodes_y*np.cos(t0) - y0

    x_undeform = r*np.cos(angles)
    y_undeform = r*np.sin(angles)

    plot_data = [exp_nodes_x, exp_nodes_y]
    plot_model = [model_nodes_x_rot, model_nodes_y_rot]

    # =======================================
    fig_name = f'Experimental VS. model fit (ellipse)'
    fh = plt.figure(fig_name)
    fh.clear()
    ax = fh.add_subplot(111)

    # ===================
    ax.plot(0.0, 0.0, 'ko', markersize=2)
    ax.plot(x0, y0, 'go', markersize=2)
    ax.plot(x_undeform, y_undeform, 'k-', linewidth=0.5, label='undeformed')
    ax.plot(plot_model[0], plot_model[1], 'b-',
                    linewidth=1.0, label='deformed (model)')
    ax.plot(plot_data[0], plot_data[1], 'ro',
                    markersize=2.0, label='deformed (data)')

    ax.legend(framealpha=0.2, bbox_to_anchor=(0.5, 0.65), loc='center',
                        fontsize=12)

    ax.set_aspect(1./ax.get_data_ratio())

    ax.set_xlabel(r"$x_1'$ [mm]", fontsize=12)
    ax.set_ylabel(r"$x_2'$ [mm]", fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)
    ax.locator_params(nbins=5)

    axins = zoomed_inset_axes(ax, 8, bbox_to_anchor=(0.52, 0.48),
                    bbox_transform=ax.transAxes, loc='upper left') # zoom = 8
    axins.plot(0.0, 0.0, 'ko', markersize=4)
    axins.plot(x0, y0, 'go', markersize=4)

    x1, x2, y1, y2 = -0.03, 0.03, -0.03, 0.03

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    axins.tick_params(axis='both', labelsize=10)
    axins.xaxis.offsetText.set_fontsize(10)
    axins.yaxis.offsetText.set_fontsize(10)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")

    plt.savefig(f'{FIGURES_FOLDER}/model_fit_ellipse.{SFF}', bbox_inches='tight')


def plot_model_fit_curve(params, exp_nodes, r, angles):

    ellip_x_data, ellip_y_data = exp_nodes

    position = []
    position_model = []

    a = params[0]
    b = params[1]
    th0 = params[2]*np.pi/180.0
    x0 = params[3]
    y0 = params[4]

    for x, y in zip(ellip_x_data, ellip_y_data):
        position.append(np.sqrt(x**2 + y**2))

    for th in angles:
        x_ = a*np.cos(th - th0)
        y_ = b*np.sin(th - th0)

        x_model = x_*np.cos(th0) - y_*np.sin(th0)
        y_model = x_*np.sin(th0) + y_*np.cos(th0)

        position_model.append(np.sqrt((x_model - x0)**2 + (y_model - y0)**2))

    fig_name = f'Deformation curve for r={r} mm'
    fh = plt.figure(fig_name)
    fh.clear()
    fig, ax = plt.subplots()

    ax.plot(angles*180/np.pi, position_model, 'b-', linewidth=2, label='model')
    ax.plot(angles*180/np.pi, position, 'ro', markersize=2, label='data')

    ax.legend(fontsize=12)
    ax.set_xlabel(r"$\theta^{\prime}$ [°]", fontsize=12)
    ax.set_ylabel(r'$r_{\mathrm{deformation}}$ [mm]', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    ax.set_aspect(1./ax.get_data_ratio())
    plt.grid(True)

    plt.savefig(f'{FIGURES_FOLDER}/mode_fit_curve.{SFF}', bbox_inches='tight')

    #==============================
    # recentred

    position_recentred = []
    position_model_recentred = []

    for th in angles:

        node_x = r*np.cos(th) + x0
        node_y = r*np.sin(th) + y0

        ux_disp_data, uy_disp_data = interpolate_r_th(node_x, node_y,
            data[0], data[1], data[2], data[3])

        ellip_x_data, ellip_y_data = ellipse_deformed(node_x, node_y,
            ux_disp_data, uy_disp_data)

        x_ = a*np.cos(th - th0)
        y_ = b*np.sin(th - th0)

        x_model = x_*np.cos(th0) - y_*np.sin(th0)
        y_model = x_*np.sin(th0) + y_*np.cos(th0)

        position_recentred.append(np.sqrt(ellip_x_data**2 + ellip_y_data**2))
        position_model_recentred.append(np.sqrt(x_model**2 + y_model**2))

    fig_name = f'Deformation curve for r={r} mm (recentred)'
    fh = plt.figure(fig_name)
    fh.clear()
    fig, ax = plt.subplots()

    p1, = ax.plot(angles*180/np.pi, position_model,
                                    'b-', linewidth=2, label='model')
    p2, = ax.plot(angles*180/np.pi, position,
                                    'ro', markersize=2, label='data')

    ax2 = ax.twiny()
    p3, = ax2.plot(angles*180/np.pi, position_model_recentred,
                    'm-', markersize=2, label='model (recentred)')
    p4, = ax2.plot(angles*180/np.pi, position_recentred,
                    'co', markersize=2, label='data (recentred)')

    ax.set_xlabel(r"$\theta^{\prime}$ [°]", fontsize=12)
    ax.set_ylabel(r'$r_{\mathrm{deformed}}$ [mm]', fontsize=12)
    ax2.set_xlabel(r"$\theta^{\prime}_r$ [°]", fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, framealpha=0.7, fontsize=12, loc='upper left')
    ax.tick_params(axis='x', colors=p1.get_color())
    ax2.tick_params(axis='x', colors=p3.get_color())
    ax2.grid(True)

    plt.savefig(f'{FIGURES_FOLDER}/mode_fit_curve_recentred.{SFF}', bbox_inches='tight')


def plot_color_map_post_identif(params, disp_x_mean_raw, disp_y_mean_raw):

    nx_point_raw = 49
    ny_point_raw = 49

    nx_points = 201
    ny_points = 201
    radius = 2.0

    x0  = params[3]
    y0  = params[4]

    xp = np.linspace(-radius, radius, nx_points)
    yp = np.linspace(-radius, radius, ny_points)

    node_x, node_y = np.meshgrid(xp, yp)

    disp_x_raw_recentred = disp_x_mean_raw + x0*1e3
    disp_y_raw_recentred = disp_y_mean_raw + y0*1e3

    ux_disp_data, uy_disp_data = interpolate_heatmap(node_x, node_y, node_x_raw,
        node_y_raw, disp_x_raw_recentred, disp_y_raw_recentred)

    ur_disp = ux_disp_data*np.cos(np.arctan2(node_y, node_x)) \
            + uy_disp_data*np.sin(np.arctan2(node_y, node_x))

    fig_name = f'Radial displacement heatmap'
    fh = plt.figure(fig_name)
    fh.clear()
    ax = fh.add_subplot(111)

    levels = MaxNLocator(nbins=25).tick_values(ur_disp.min(), ur_disp.max())

    # cf = ax.contourf(node_x, node_y, ur_disp, levels=levels, cmap='jet')
    cf = ax.pcolormesh(node_x, node_y, ur_disp, cmap='jet')
    clb = plt.colorbar(cf, ax=ax)
    # ax.plot(x0, y0, 'k+', markersize=2)
    # ax.plot(0, 0, 'ko', markersize=2)

    clb.ax.set_title(r'$u_r$ [$\mu$m]', fontsize=12)

    ax.set_xlabel(r'$x_1^{\prime}$ [mm]', fontsize=12)
    ax.set_ylabel(r'$x_2^{\prime}$ [mm]', fontsize=12)

    plt.savefig(f'{FIGURES_FOLDER}/color_map.{SFF}', bbox_inches='tight')

# ==============================================================
# ==============================================================
### Experimental data

DIC_bool = input('Perform the DIC process [Y/n]: ')
if DIC_bool == ('Y' or 'y'):
    DIC_PROCESS = True
    SAVE_VIDEO_FRAMES = True
    DISP_FIELD_IMAGES = True
else:
    print('You choosed to not use DIC. Warning! the parameter identification \
        will be processed on the existing DIC files in the folder.')
    DIC_PROCESS = False
    SAVE_VIDEO_FRAMES = False
    DISP_FIELD_IMAGES = False

if DIC_PROCESS:
    import pydic
    current_dir = os.getcwd()
    dir = os.path.join(current_dir, f'{SOURCE_FOLDER}')
    pydic_methods = os.path.abspath('./pydic_methods.py')
    pydic.suction_dic(dir, pydic_methods, SAVE_VIDEO_FRAMES, DISP_FIELD_IMAGES)
    os.chdir(current_dir)

if PLOT_MODEL_FIT or PLOT_DISP_FIELD or PLOT_DEFORMATION:
    make_new_directory(f'{FIGURES_FOLDER}')

theta_input = float(input('Insert the initial anisotropy angle (in degrees):\
(for demo type 45): '))
initial_set = {
    #WARNING: avoid a = b
    'a': 1.0, #(MPa)
    'b': 1.1, #(MPa)
    'theta0': theta_input, #[°]
    'x0': 0.0,
    'y0': 0.0,
    }

R = float(input('Insert the radius of COI (in mm): (for demo type 1): '))
num_points = 100

print('Importing experimental displacement data and grid nodes.')
import data_preparation

export = data_preparation.export_data(FRAME_NUM, SOURCE_FOLDER)

node_x_raw = np.array(export['node_x'], np.float64)
node_y_raw = np.array(export['node_y'], np.float64)

disp_x_mean_raw = np.array(export['disp_x_mean'], np.float64)
disp_y_mean_raw = np.array(export['disp_y_mean'], np.float64)

data = [node_x_raw, node_y_raw, disp_x_mean_raw, disp_y_mean_raw]

## Organizing experimental data
points = []
for x, y in zip(node_x_raw, node_y_raw):
    points.append([x, y])
nodes = np.array(points, np.float64)

data_reponse = [] # Measured experimental data on grid points
for x, y in zip(disp_x_mean_raw, disp_y_mean_raw):
    data_reponse.append([x*1e-3, y*1e-3])
disp = np.array(data_reponse, np.float64)

angles = np.linspace(0, 2*np.pi, num_points)
plot_data = []

node_x = R*np.cos(angles)
node_y = R*np.sin(angles)

ux_disp_data, uy_disp_data = interpolate_r_th(node_x, node_y, node_x_raw,
        node_y_raw, disp_x_mean_raw, disp_y_mean_raw)

ellip_x_data, ellip_y_data = ellipse_deformed(node_x, node_y,
        ux_disp_data, uy_disp_data)

# =================================================================
if PLOT_DISP_FIELD:

    fig_name = f'Experimental displacement field for p = {pressure} mbar'
    fh = plt.figure(fig_name)
    fh.clear()
    ax = fh.add_subplot(111)

    # ================
    x = node_x_raw
    y = node_y_raw
    u = disp_x_mean_raw
    v = disp_y_mean_raw

    ax.quiver(x, y, u, v, color='k', linewidth=5)
    plt.gca().set_aspect("equal")
    ax.set_xlabel(r"$x_1'$ [mm]", fontsize=12)
    ax.set_ylabel(r"$x_2'$ [mm]", fontsize=12)
    ax.tick_params('both', labelsize=12)

    plt.savefig(f'{FIGURES_FOLDER}/field_exp.{SFF}', bbox_inches='tight')


#==========================================================
if PLOT_DEFORMATION:

    fig_name = f'Deformed and initial configuration'
    fh = plt.figure(fig_name)
    fh.clear()
    ax = fh.add_subplot(111)

    ax.plot(node_x, node_y, 'b-', linewidth=1, label='undeformed')
    ax.plot(ellip_x_data, ellip_y_data,
                                    'k-.', markersize=1, label='deformed')
    ax.legend(framealpha=0.2)
    plt.gca().set_aspect("equal")
    ax.set_xlabel(r"$x_1'$ [mm]", fontsize=12)
    ax.set_ylabel(r"$x_2'$ [mm]", fontsize=12)
    ax.tick_params('both', labelsize=12)
    plt.grid(True)

    plt.savefig(f'{FIGURES_FOLDER}/deformed_ellipse.{SFF}', bbox_inches='tight')


# =================================================================
# =================================================================
### Inverse identification

if IDENTIFY_PARAMETERS:
    print('Parameter identification...')

    exp_nodes = []
    for x, y in zip(ellip_x_data, ellip_y_data):
        exp_nodes.append([x, y])
    exp_nodes = np.array(exp_nodes, np.float64)

    params, err_rel, deter_coeff, num_iter = newton.inverse_solve( initial_set,
        exp_nodes, angles, R)

    print(f'Converged parameters are:')
    print(f'Minor semi-axis a (mm): {params[0][0]}')
    print(f'Major semi-axis b (mm): {params[1][0]}')
    print(f'Anisotropy angle phi (deg): {params[2][0]}')
    print(f'Center shift X1-coord (mm): {params[3][0]}')
    print(f'Center shift X2-coord (mm): {params[4][0]}')
    print(f'Correlation coefficient is: {np.sqrt(deter_coeff)[0]}')

    if params[0] > params[1]:
        print("'a' parameter is not smaller than 'b'! The result is not valid. "+\
            "Recalcultes with a different initial anisotropy angle value")

    if params[0] < 0 or params[1] < 0:
        print("'a' and/or 'b' are not positive ! The result is not valid. " +\
            "Recalcultes with a different initial anisotropy angle value")

if SAVE_IDENTIFIED_PARAMETERS:
    RESULT_FILE = open(FIGURES_FOLDER + '/identification_results.txt', 'wt')
    RESULT_FILE.write(f'Ring suction pressure: {pressure} mbar \n')
    RESULT_FILE.write(f'Radius of the circle of interest: {R} mm \n')
    RESULT_FILE.write(f'a = {params[0][0]} mm \n')
    RESULT_FILE.write(f'b = {params[1][0]} mm \n')
    RESULT_FILE.write(f'phi = {params[2][0]} deg \n')
    RESULT_FILE.write(f'x1,0 = {params[3][0]} mm \n')
    RESULT_FILE.write(f'x2,0 = {params[4][0]} mm \n')
    RESULT_FILE.write(f'Correlation coefficient is: {np.sqrt(deter_coeff)[0]}')
    RESULT_FILE.close()

# ==============================================================
if PLOT_MODEL_FIT:

    def long_process():
        plot_model_fit_curve(params, [ellip_x_data, ellip_y_data], R, angles)
        plot_model_fit_ellipse(params, [ellip_x_data, ellip_y_data], R, angles)
        plot_axis_rotation(params, [ellip_x_data, ellip_y_data], R, angles)
        plot_color_map_post_identif(params, disp_x_mean_raw, disp_y_mean_raw)

    thread = threading.Thread(target=long_process)
    thread.start()
    for c in itertools.cycle(['|', '/', '-', '\\']):
        sys.stdout.write('\rPloting and exporting figures' + c)
        sys.stdout.flush()
        time.sleep(0.1)
        if not thread.is_alive():
            break
    sys.stdout.write('\rAll plots were saved !         \n')
