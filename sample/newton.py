'''
This program identify material paramters basing on Newton-Gauss LSQ method.
The inputs are:
- Initial guess
- Experimental data

The outputs are:
- Idenfied parameters

This inverse solver has been implemented especially for one specific model:
2D otrhotropic linear elasticity of circular domain, with the following boundary
conditions:
    -   null displacement in the center.
    -   constant negative pressure on the circonference.

=============================
Author: @aflah elouneg
Version 2.0:    08/07/2021
=============================
'''

import numpy as np
import scipy.linalg as linalg


def cost_gradient(param, node_coord, angle):
    ''' compute the gradient array of the objective function J regarding the
    material parameters on every node
    J = 1/2*J_^2 = [sqrt((x - x0)**2 + (y - y0)**2) - sqrt(xi**2 + yi**2)]^2
    {x, y}: model point coordinates
    {xi, yi}: experimental point coordinates
    '''

    a   = param[0]
    b   = param[1]
    t0  = param[2]
    x0  = param[3]
    y0  = param[4]

    xi, yi    = node_coord

    n_param = 5

    gradient_array = np.zeros((n_param, 1), np.float64)

    t   = np.arctan2(yi, xi)

    x_ = a*np.cos(t - t0)
    y_ = b*np.sin(t - t0)

    dx_dt0 = -(-a*np.sin(t - t0))
    dy_dt0 = -(b*np.cos(t - t0))

    x = x_*np.cos(t0) - y_*np.sin(t0)
    y = x_*np.sin(t0) + y_*np.cos(t0)

    dxda = np.cos(t - t0)*np.cos(t0)
    dyda = np.cos(t - t0)*np.sin(t0)

    dxdb = -np.sin(t - t0)*np.sin(t0)
    dydb = np.sin(t - t0)*np.cos(t0)

    dxdt0 = dx_dt0*np.cos(t0) - x_*np.sin(t0) - dy_dt0*np.sin(t0) - y_*np.cos(t0)
    dydt0 = dx_dt0*np.sin(t0) + x_*np.cos(t0) + dy_dt0*np.cos(t0) - y_*np.sin(t0)

    J_ = np.sqrt((x - x0)**2 + (y - y0)**2) - np.sqrt(xi**2 + yi**2)

    xr = x - x0
    yr = y - y0

    dJ_da  = (2*dxda*xr + 2*dyda*yr)*(0.5)*(xr**2 + yr**2)**(-0.5)
    dJ_db  = (2*dxdb*xr + 2*dydb*yr)*(0.5)*(xr**2 + yr**2)**(-0.5)
    dJ_dt0 = (2*dxdt0*xr + 2*dydt0*yr)*(0.5)*(xr**2 + yr**2)**(-0.5)
    dJ_dx0 = -2*xr*(0.5)*(xr**2 + yr**2)**(-0.5)
    dJ_dy0 = -2*yr*(0.5)*(xr**2 + yr**2)**(-0.5)

    dJda  = dJ_da*J_
    dJdb  = dJ_db*J_
    dJdt0 = dJ_dt0*J_
    dJdx0 = dJ_dx0*J_
    dJdy0 = dJ_dy0*J_

    # input('pause')

    gradient_array[0] = dJda
    gradient_array[1] = dJdb
    gradient_array[2] = dJdt0
    gradient_array[3] = dJdx0
    gradient_array[4] = dJdy0

    return gradient_array


def cost_hessian(param, node_coord, angle):
    ''' compute the hessian matrix of the objective function J regarding the
    material parameters on every node
    J = 1/2*J_^2 = [sqrt((x - x0)**2 + (y - y0)**2) - sqrt(xi**2 + yi**2)]^2
    {x, y}: model point coordinates
    {xi, yi}: experimental point coordinates
    '''

    a   = param[0]
    b   = param[1]
    t0  = param[2]
    x0  = param[3]
    y0  = param[4]

    xi, yi    = node_coord

    n_param = 5

    hessian_matrix = np.zeros((n_param, n_param), np.float64)

    t   = np.arctan2(yi, xi)

    x_ = a*np.cos(t - t0)
    y_ = b*np.sin(t - t0)

    dx_dt0 = -(-a*np.sin(t - t0))
    dy_dt0 = -(b*np.cos(t - t0))

    d2x_dt02 = -a*np.cos(t - t0)
    d2y_dt02 = -b*np.sin(t - t0)

    x = x_*np.cos(t0) - y_*np.sin(t0)
    y = x_*np.sin(t0) + y_*np.cos(t0)

    dxda = np.cos(t - t0)*np.cos(t0)
    dyda = np.cos(t - t0)*np.sin(t0)

    dxdb = -np.sin(t - t0)*np.sin(t0)
    dydb = np.sin(t - t0)*np.cos(t0)

    dxdt0 = (dx_dt0 - y_)*np.cos(t0) - (x_ + dy_dt0)*np.sin(t0)
    dydt0 = (dx_dt0 - y_)*np.sin(t0) + (x_ + dy_dt0)*np.cos(t0)

    d2xda2 = 0.0
    d2yda2 = 0.0

    d2xdb2 = 0.0
    d2ydb2 = 0.0

    d2xdt02 = (d2x_dt02 - dy_dt0)*np.cos(t0) - (dx_dt0 - y_)*np.sin(t0) -\
        (dx_dt0 + d2y_dt02)*np.sin(t0) - (x_ + dy_dt0)*np.cos(t0)
    d2ydt02 = (d2x_dt02 - dy_dt0)*np.sin(t0) + (dx_dt0 - y_)*np.cos(t0) +\
        (dx_dt0 + d2y_dt02)*np.cos(t0) - (x_ + dy_dt0)*np.sin(t0)

    d2xdadt0 = -(-np.sin(t - t0))*np.cos(t0) - np.cos(t - t0)*np.sin(t0)
    d2ydadt0 = -(-np.sin(t - t0))*np.sin(t0) + np.cos(t - t0)*np.cos(t0)
    d2xdbdt0 = -(-np.cos(t - t0))*np.sin(t0) - np.sin(t - t0)*np.cos(t0)
    d2ydbdt0 = -(np.cos(t - t0))*np.cos(t0) - (np.sin(t - t0)*np.sin(t0))

    d2xdadx0 = 0.0
    d2ydadx0 = 0.0
    d2xdbdx0 = 0.0
    d2ydbdx0 = 0.0

    d2xdady0 = 0.0
    d2ydady0 = 0.0
    d2xdbdy0 = 0.0
    d2ydbdy0 = 0.0

    d2xdadb = 0.0
    d2ydadb = 0.0

    J_ = np.sqrt((x - x0)**2 + (y - y0)**2) - np.sqrt(xi**2 + yi**2)

    xr = x - x0
    yr = y - y0

    dxrdx0 = -1.0
    dyrdy0 = -1.0
    d2xrdx02 = 0.0
    d2yrdy02 = 0.0

    dJ_da  = (2*dxda*xr + 2*dyda*yr)*(0.5)*(xr**2 + yr**2)**(-0.5)
    dJ_db  = (2*dxdb*xr + 2*dydb*yr)*(0.5)*(xr**2 + yr**2)**(-0.5)
    dJ_dt0 = (2*dxdt0*xr + 2*dydt0*yr)*(0.5)*(xr**2 + yr**2)**(-0.5)
    dJ_dx0 = -2*xr*(0.5)*(xr**2 + yr**2)**(-0.5)
    dJ_dy0 = -2*yr*(0.5)*(xr**2 + yr**2)**(-0.5)

    d2J_da2  = (2*d2xda2*xr + 2*d2yda2*yr + 2*dxda**2 + 2*dyda**2)*\
        (0.5)*(xr**2 + yr**2)**(-0.5) +\
        (2*dxda*xr + 2*dyda*yr)**2*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)
    d2J_db2  = (2*d2xdb2*xr + 2*d2ydb2*yr + 2*dxdb**2 + 2*dydb**2)*\
        (0.5)*(xr**2 + yr**2)**(-0.5) +\
        (2*dxdb*xr + 2*dydb*yr)**2*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)
    d2J_dt02 = (2*d2xdt02*xr + 2*dxdt0**2 + 2*d2ydt02*yr + 2*dydt0**2)*\
        (0.5)*(xr**2 + yr**2)**(-0.5) +\
        (2*dxdt0*xr + 2*dydt0*yr)**2*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)
    d2J_dx02 = 2*0.5*(xr**2 + yr**2)**(-0.5) + 4*xr**2*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)
    d2J_dy02 = 2*0.5*(xr**2 + yr**2)**(-0.5) + 4*yr**2*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)

    d2J_dadb = (2*d2xdadb*xr + 2*d2ydadb*yr + 2*dxda*dxdb + 2*dyda*dydb)*\
        (0.5)*(xr**2 + yr**2)**(-0.5) + (2*dxda*xr + 2*dyda*yr)*\
        (2*dxdb*xr + 2*dydb*yr)*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)
    d2J_dadt0 = (2*d2xdadt0*xr + 2*d2ydadt0*yr + 2*dxda*dxdt0 + 2*dyda*dydt0)*\
        (0.5)*(xr**2 + yr**2)**(-0.5) +\
        (2*dxda*xr + 2*dyda*yr)*(2*dxdt0*xr + 2*dydt0*yr)*(0.5)*(-0.5)*\
        (xr**2 + yr**2)**(-1.5)
    d2J_dadx0 = -2*dxda*(0.5)*(xr**2 + yr**2)**(-0.5) -\
            2*xr*(2*dxda*xr + 2*dyda*yr)*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)
    d2J_dady0 = -2*dyda*(0.5)*(xr**2 + yr**2)**(-0.5) -\
            2*yr*(2*dxda*xr + 2*dyda*yr)*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)

    d2J_dbdt0 = (2*d2xdbdt0*xr + 2*d2ydbdt0*yr + 2*dxdb*dxdt0 + 2*dydb*dydt0)*\
        (0.5)*(xr**2 + yr**2)**(-0.5) +\
        (2*dxdb*xr + 2*dydb*yr)*(2*dxdt0*xr + 2*dydt0*yr)*(0.5)*(-0.5)*\
        (xr**2 + yr**2)**(-1.5)

    d2J_dbdx0  = -2*dxdb*xr*(0.5)*(xr**2 + yr**2)**(-0.5) -\
            2*xr*(2*dxdb*xr + 2*dydb*yr)*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)
    d2J_dbdy0  = -2*dydb*yr*(0.5)*(xr**2 + yr**2)**(-0.5) -\
            2*yr*(2*dxdb*xr + 2*dydb*yr)*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)

    d2J_dt0dx0 = -2*dxdt0*(0.5)*(xr**2 + yr**2)**(-0.5) -\
            2*xr*(2*dxdt0*xr + 2*dydt0*yr)*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)
    d2J_dt0dy0 = -2*dydt0*(0.5)*(xr**2 + yr**2)**(-0.5) -\
            2*yr*(2*dxdt0*xr + 2*dydt0*yr)*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)

    d2J_dx0dy0 = 4*xr*yr*(0.5)*(-0.5)*(xr**2 + yr**2)**(-1.5)

    # ====================

    d2Jda2  = d2J_da2*J_ + dJ_da**2
    d2Jdb2  = d2J_db2*J_ + dJ_db**2
    d2Jdt02 = d2J_dt02*J_ + dJ_dt0**2
    d2Jdx02 = d2J_dx02*J_ + dJ_dx0**2
    d2Jdy02 = d2J_dy02*J_ + dJ_dy0**2

    d2Jdadb  = d2J_dadb*J_ + dJ_da*dJ_db
    d2Jdadt0 = d2J_dadt0*J_ + dJ_da*dJ_dt0
    d2Jdadx0 = d2J_dadx0*J_ + dJ_da*dJ_dx0
    d2Jdady0 = d2J_dady0*J_ + dJ_da*dJ_dy0

    d2Jdbdt0 = d2J_dbdt0*J_ + dJ_db*dJ_dt0
    d2Jdbdx0 = d2J_dbdx0*J_ + dJ_db*dJ_dx0
    d2Jdbdy0 = d2J_dbdy0*J_ + dJ_db*dJ_dy0

    d2Jdt0dx0 = d2J_dt0dx0*J_ + dJ_dt0*dJ_dx0
    d2Jdt0dy0 = d2J_dt0dy0*J_ + dJ_dt0*dJ_dy0

    d2Jdx0dy0 = d2J_dx0dy0*J_ + dJ_dx0*dJ_dy0
    # ====================

    hessian_matrix[0, 0] = d2Jda2
    hessian_matrix[0, 1] = d2Jdadb
    hessian_matrix[0, 2] = d2Jdadt0
    hessian_matrix[0, 3] = d2Jdadx0
    hessian_matrix[0, 4] = d2Jdady0

    hessian_matrix[1, 0] = d2Jdadb
    hessian_matrix[1, 1] = d2Jdb2
    hessian_matrix[1, 2] = d2Jdbdt0
    hessian_matrix[1, 3] = d2Jdbdx0
    hessian_matrix[1, 4] = d2Jdbdy0

    hessian_matrix[2, 0] = d2Jdadt0
    hessian_matrix[2, 1] = d2Jdbdt0
    hessian_matrix[2, 2] = d2Jdt02
    hessian_matrix[2, 3] = d2Jdt0dx0
    hessian_matrix[2, 4] = d2Jdt0dy0

    hessian_matrix[3, 0] = d2Jdadx0
    hessian_matrix[3, 1] = d2Jdbdx0
    hessian_matrix[3, 2] = d2Jdt0dx0
    hessian_matrix[3, 3] = d2Jdx02
    hessian_matrix[3, 4] = d2Jdx0dy0

    hessian_matrix[4, 0] = d2Jdady0
    hessian_matrix[4, 1] = d2Jdbdy0
    hessian_matrix[4, 2] = d2Jdt0dy0
    hessian_matrix[4, 3] = d2Jdx0dy0
    hessian_matrix[4, 4] = d2Jdy02

    return hessian_matrix

def residual(param, node_coord, angle, R):
    ''' compute the hessian matrix of the objective function J regarding the
    material parameters on every node'''

    a   = param[0]
    b   = param[1]
    t0  = param[2]
    x0  = param[3]
    y0  = param[4]

    xi, yi    = node_coord

    t   = np.arctan2(yi, xi)

    x_ = a*np.cos(t - t0)
    y_ = b*np.sin(t - t0)

    x = x_*np.cos(t0) - y_*np.sin(t0)
    y = x_*np.sin(t0) + y_*np.cos(t0)

    J_ = np.sqrt((x - x0)**2 + (y - y0)**2) - np.sqrt(xi**2 + yi**2)

    Y_exp = np.sqrt(xi**2 + yi**2)
    SSE_ = J_**2

    return np.abs(J_), np.abs(np.sqrt(xi**2 + yi**2) - R), Y_exp, SSE_


def inverse_solve(initial, nodes, angles, R):

    rel_tol = 1e-9
    iter_max = 50

    CONVERGED = False

    a   = initial['a']
    b   = initial['b']
    t0  = initial['theta0']
    x0  = initial['x0']
    y0  = initial['y0']

    m = np.array([[a], [b], [t0*np.pi/180.0], [x0], [y0]])

    iter = 0

    n_param = 5

    while not CONVERGED and iter <= iter_max:

        iter += 1

        cumul_gradient = np.zeros((n_param, 1), np.float64)
        cumul_hessian  = np.zeros((n_param, n_param), np.float64)
        cumul_cost     = 0
        cumul_disp     = 0
        SSE = 0 # Sum of Squares Due to Error
        Y_exp = [] # Sum of Squares Total

        for node, angle in zip(nodes, angles):
            cumul_gradient += cost_gradient(m, node, angle)
            cumul_hessian  += cost_hessian(m, node, angle)
            comul_cost_, cumul_disp_, Y_exp_, SSE_ = residual(m, node, angle, R)
            cumul_cost += comul_cost_
            cumul_disp += cumul_disp_
            SSE += SSE_
            Y_exp.append(Y_exp_)

        R2 = 1 - SSE/(np.sum((np.array(Y_exp) - np.mean(Y_exp))**2))
        dm = -np.dot(np.linalg.inv(cumul_hessian), cumul_gradient)

        m += dm

        angle_ = m[2]%(2*np.pi)
        if angle_ >= np.pi:
            angle_ = angle_ - 2*np.pi
        m[2] = angle_

        if m[2] < -np.pi/2:
            m[2] += np.pi

        if m[2] > np.pi/2:
            m[2] += -np.pi

        if np.all(np.abs(dm)[:] < rel_tol):
            CONVERGED = True

        # input('pause')
    resid = cumul_cost/cumul_disp
    m[2] = m[2]*180.0/np.pi
    return m, resid, R2, iter

def constrain_dm(dm, m):
    '''Constrain the change (max L2 change).'''
    delta = linalg.norm(dm)/linalg.norm(m)
    if delta > 0.5:
        dm *= 0.5/delta
    return dm
