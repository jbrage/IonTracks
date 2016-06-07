import cython
# Declarations exclusively to either pure Python or Cython
if not cython.compiled:
    # No-op decorators for Cython compiler directives
    def dummy_decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # Called as @dummy_decorator. Return function
            return args[0]
        else:
            # Called as @dummy_decorator(*args, **kwargs).
            # Return decorator
            return dummy_decorator
    # Already builtin: cfunc, inline, locals, returns
    for directive in ('boundscheck','cdivision','initializedcheck','wraparound','header'):
        setattr(cython, directive, dummy_decorator)
    # Dummy Cython functions
    for func in ('address',):
        setattr(cython, func, lambda _: _)
else:
    # Lines in triple quotes will be executed in .pyx files
    """
    # Get full access to all of Cython
    cimport cython
    """
# Seperate but equivalent imports and
# definitions in pure Python and Cython
if not cython.compiled:
    # Mathematical constants and functions
    from numpy import (sin, cos, tan, arcsin, arccos, arctan,
                       sinh, cosh, tanh, arcsinh, arccosh, arctanh,
                       exp, log, log2, log10,
                       sqrt,pi
                       )
    from math import erfc
else:
    # Lines in triple quotes will be executed in .pyx files.
    """
    # Mathematical constants and functions
    from libc.math cimport (M_PI as pi,
                            sin, cos, tan,
                            asin as arcsin,
                            acos as arccos,
                            atan as arctan,
                            sinh, cosh, tanh,
                            asinh as arcsinh,
                            acosh as arccosh,
                            atanh as arctanh,
                            exp, log, log2, log10,
                            sqrt, erfc
                            )
    """

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.gridspec as gridspec
import pylab as pl
import math


@cython.ccall
@cython.locals( # variables
                track_radii='double[::1]',
                track_LETs='double[::1]',
                parameter_list='list',
                SHOW_PLOT='bint',
                #locals
                )
def distribute_and_evolve(track_radii,track_LETs,parameter_list,SHOW_PLOT):
    return f(track_radii,track_LETs,parameter_list,SHOW_PLOT)


@cython.header( # variables
                track_radii='double[::1]',
                track_LETs='double[::1]',
                parameter_list='list',
                SHOW_PLOT='bint',
                # arrays
                positive_array='double[:,:,::1]', negative_array='double[:,:,::1]',positive_array_temp='double[:,:,::1]',
                negative_array_temp='double[:,:,::1]',positive_array_slice='double[:,::1]',
                # load variables
                ion_diff='double',ion_mobility='double',Efield='double',dx='double',dy='double',dz='double',W='double',
                no_z_electrode='int', d='double',theta='double',alpha='double',dt='double',
                no_x='size_t',no_y='size_t',no_z='size_t',no_z_with_buffer='size_t',
                # indices and lengths
                nx='size_t',ny='size_t',nz='size_t',i='size_t',j='size_t',k='size_t',
                mid_x_array='size_t',mid_y_array='size_t',mid_z_array='size_t',N0='double',b='double',front_factor='double',distance_from_center='double',
                # von Neumann analysis
                sx='double', sy='double', sz='double', cx='double', cy='double', cz='double',von_neumann_expression='bint',
                # Lax-Wendroff scheme
                positive_temp_entry='double',negative_temp_entry='double',
                recomb_temp='double',no_recombined_charge_carriers='double',

                update_figure_step='size_t',
                no_initialised_charge_carriers='double',ion_density='double',computation_time_steps='int',time_step='int',no_figure_updates='size_t',
                )

def f(track_radii,track_LETs,parameter_list,SHOW_PLOT):

    #initialize the simulation parameters:
    ion_diff    = parameter_list[0] # [UNITS] ion diffusivity
    ion_mobility= parameter_list[1] # [UNITS] ion mobility
    Efield      = parameter_list[2] # [V/cm] magnitude of the electric field
    dx          = parameter_list[3] # [cm] distance between two neighbouring voxels (x-direction)
    dy          = parameter_list[4] # [cm] distance between two neighbouring voxels (y-direction)
    dz          = parameter_list[5] # [cm] distance between two neighbouring voxels (z-direction)
    theta       = parameter_list[6] # [rad] angle between electric field and the ion track(s)
    alpha       = parameter_list[7] # [cm^3/s] recombination constant
    no_x        = parameter_list[8] # number of elements in the x-direction
    no_y        = parameter_list[9]# number of elements in the y-direction
    no_z        = parameter_list[10]# number of elements in the z-direction
    no_z_electrode   = parameter_list[11] # remove?
    d           = parameter_list[12] # [cm] # electrode gap
    W           = parameter_list[13] # [UNITS] energy required to form a pair of ions

    # total length, i.e. electrode gap plus the two electrode buffers
    no_z_with_buffer = 2*no_z_electrode + no_z

    # number of times the figure is updated
    no_figure_updates = 5

    # preallocate arrays
    positive_array=np.zeros((no_x,no_y,no_z+2*no_z_electrode))
    positive_array_temp=np.zeros((no_x,no_y,no_z+2*no_z_electrode))
    negative_array_temp = np.zeros((no_x, no_y, no_z + 2 * no_z_electrode))
    no_recombined_charge_carriers = 0.
    no_initialised_charge_carriers = 0.

    # find the middle of the arrays (mainly for the Gaussian distribution and updating the figures)
    mid_x_array=int(no_x/2.)
    mid_y_array=mid_x_array
    mid_z_array=int(no_z/2.)

    # ensure the resolution is the same in all directions for the the approximations below
    assert(dx==dy and dy==dz)

    #initialize a Gaussian track carrier distribution around a track in the center of the array
    N0 = track_LETs[0]/W # Linear charge carrier density
    b = track_radii[0] # Gaussian track radius
    front_factor = N0/(pi*b**2)

    # do not initialize charge carriers in the electrodes arrays
    for k in range(no_z_electrode,no_z+no_z_electrode):
        for i in range(no_x):
            for j in range(no_y):
                distance_from_center=sqrt((i-mid_x_array)**2 + (j-mid_y_array)**2)*dx
                ion_density=front_factor*exp(-distance_from_center**2/b**2)
                positive_array[i,j,k]=ion_density
                no_initialised_charge_carriers+=ion_density

    negative_array=deepcopy(np.asarray(positive_array,dtype='double'))

    # for the colorbars
    MINVAL = 0.
    MAXVAL = front_factor

    if SHOW_PLOT:
        plt.close('all')
        fig = plt.figure()
        plt.ion()
        gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

        # define locations of subfigures in the grid
        ax1 = plt.subplot2grid((5,3),(0,0),rowspan=4)
        ax2 = plt.subplot2grid((5,3),(0,1),rowspan=4)
        ax3 = plt.subplot2grid((5,3),(0,2),rowspan=4)

        ax1.imshow(np.asarray(positive_array[mid_x_array,:,:]).transpose(), vmin=MINVAL, vmax=MAXVAL)
        ax2.imshow(positive_array[:,:,mid_y_array], vmin=MINVAL, vmax=MAXVAL)
        figure3 = ax3.imshow(positive_array[:,:,mid_x_array], vmin=MINVAL, vmax=MAXVAL)

        # adjust the 3 subfigures
        ax1.set_aspect('equal'); ax1.axis('off')
        ax2.set_aspect('equal'); ax2.axis('off')
        ax3.set_aspect('equal'); ax3.axis('off')

        # fix colorbar
        cbar_ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.03])
        cb = fig.colorbar(figure3, cax=cbar_ax1, orientation='horizontal', label="Charge carrier density [per cm^3]")
        cb.set_clim(vmin=MINVAL, vmax=MAXVAL)

    # include space charge screenings of the electric field
    # if space_charge_screening_effects > 0:


    # find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
    # decreases and eventually damps out
    dt = 1.
    von_neumann_expression = False
    while not von_neumann_expression:
        dt /= 1.01
        # as defined in the Deghan (2004) paper
        sx = ion_diff*dt/(dx**2)
        sy = ion_diff*dt/(dy**2)
        sz = ion_diff*dt/(dz**2)

        cx = ion_mobility*Efield*dt/dx*sin(theta)
        cy = 0
        cz = ion_mobility*Efield*dt/dz*cos(theta)

        # check von Neumann's criterion
        von_neumann_expression = (2*(sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1 and cx**2*cy**2*cz**2 <= 8*sx*sy*sz)

    # calculate the number of step required to drag the two charge carrier distributions apart
    computation_time_steps=int(d/(2.*ion_mobility*Efield*dt))

    # one time step at a time
    for time_step in range(computation_time_steps):

        # update the figure
        if SHOW_PLOT:
            update_figure_step=int(computation_time_steps/no_figure_updates)
            if time_step % update_figure_step == 0:
                cb.set_clim(vmin=MINVAL, vmax=MAXVAL)

                ax1.imshow(np.asarray(negative_array[:,mid_y_array,:],dtype=np.double).transpose(),vmin=MINVAL,vmax=MAXVAL)
                ax2.imshow(np.asarray(positive_array[:,mid_y_array,:],dtype=np.double).transpose(),vmin=MINVAL,vmax=MAXVAL)
                ax3.imshow(np.asarray(positive_array[:,:,mid_z_array],dtype=np.double),vmin=MINVAL,vmax=MAXVAL)

                plt.pause(1e-3)

        # calculate the new densities and store them in temporary arrays
        for i in range(1,no_x-1):
            for j in range(1,no_y-1):
                for k in range(1,no_z_with_buffer-1):
                    # using the Lax-Wendroff scheme
                    positive_temp_entry = (sz+cz*(cz+1.)/2.)*positive_array[i,j,k-1]
                    positive_temp_entry += (sz+cz*(cz-1.)/2.)*positive_array[i,j,k+1]

                    positive_temp_entry += (sy+cy*(cy+1.)/2.)*positive_array[i,j-1,k]
                    positive_temp_entry += (sy+cy*(cy-1.)/2.)*positive_array[i,j+1,k]

                    positive_temp_entry += (sx+cx*(cx+1.)/2.)*positive_array[i-1,j,k]
                    positive_temp_entry += (sx+cx*(cx-1.)/2.)*positive_array[i+1,j,k]

                    positive_temp_entry += (1.- cx*cx - cy*cy - cz*cz - 2.*(sx+sy+sz))*positive_array[i,j,k]

                    # same for the negative charge carriers
                    negative_temp_entry = (sz+cz*(cz+1.)/2.)*negative_array[i,j,k+1]
                    negative_temp_entry += (sz+cz*(cz-1.)/2.)*negative_array[i,j,k-1]

                    negative_temp_entry += (sy+cy*(cy+1.)/2.)*negative_array[i,j+1,k]
                    negative_temp_entry += (sy+cy*(cy-1.)/2.)*negative_array[i,j-1,k]

                    negative_temp_entry += (sx+cx*(cx+1.)/2.)*negative_array[i+1,j,k]
                    negative_temp_entry += (sx+cx*(cx-1.)/2.)*negative_array[i-1,j,k]

                    negative_temp_entry += (1. - cx*cx - cy*cy - cz*cz - 2.*(sx+sy+sz))*negative_array[i,j,k]

                    # the recombination part
                    recomb_temp = alpha*positive_array[i,j,k]*negative_array[i,j,k]*dt
                    # the total recombination
                    no_recombined_charge_carriers += recomb_temp

                    # positive array
                    positive_array_temp[i,j,k] = positive_temp_entry - recomb_temp
                    # negative array
                    negative_array_temp[i,j,k] = negative_temp_entry - recomb_temp

        # update the positive and negative arrays
        for i in range(1,no_x-1):
            for j in range(1,no_y-1):
                for k in range(1,no_z_with_buffer-1):
                    positive_array[i,j,k] = positive_array_temp[i,j,k]
                    negative_array[i,j,k] = negative_array_temp[i,j,k]

    return [no_initialised_charge_carriers,no_recombined_charge_carriers]