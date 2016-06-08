import cython
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.gridspec as gridspec
import numpy.random as rnd

# Declarations exclusively to either pure Python or Cython
# Lines in triple quotes will be executed in .pyx files

# Get full access to all of Cython
cimport cython

# Seperate but equivalent imports and
# definitions in pure Python and Cython
# Lines in triple quotes will be executed in .pyx files.

# Mathematical constants and functions
from libc.math cimport (M_PI as pi, sin, cos, tan, asin as arcsin, acos as arccos, atan as arctan, sinh, cosh, tanh, asinh as arcsinh, acosh as arccosh, atanh as arctanh, exp, log, log2, log10, sqrt, erfc )

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


@cython.cfunc
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.locals( # variables
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
                # initialise charge carriers
                x_coordinates_ALL='double[::1]',y_coordinates_ALL='double[::1]',number_of_tracks='size_t',inner_radius='size_t',outer_radius='size_t',
                x='double',y='double',my_counter='size_t',MAXVAL='double',MINVAL='double',

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
    no_y        = parameter_list[9]  # number of elements in the y-direction
    no_z        = parameter_list[10]  # number of elements in the z-direction
    no_z_electrode   = parameter_list[11] # remove?
    d           = parameter_list[12] # [cm] # electrode gap
    W           = parameter_list[13] # [UNITS] energy required to form a pair of ions

    # total length, i.e. electrode gap plus the two electrode buffers
    no_z_with_buffer = 2*no_z_electrode + no_z

    # number of times the figure is updated
    no_figure_updates = 5

    number_of_iterations = 1e4
    x_coordinates_ALL = rnd.uniform(0,1,number_of_iterations*10)*no_x
    y_coordinates_ALL = rnd.uniform(0,1,number_of_iterations*10)*no_y


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

    my_counter = 0
    outer_radius = int(no_x/2.)
    inner_radius = outer_radius-10

    # for the colorbars
    MINVAL = 0.
    MAXVAL = 0.

    number_of_tracks = len(track_LETs)
    for track_number in range(number_of_tracks):

        N0 = track_LETs[track_number]/W     # Linear charge carrier density
        b = track_radii[track_number]       # Gaussian track radius
        front_factor = N0/(pi*(b*b))

        my_counter += 1
        x = x_coordinates_ALL[my_counter]
        y = y_coordinates_ALL[my_counter]

        # faster than just
        while sqrt(((x - mid_x_array)   *(x - mid_x_array)   )+ ((y - mid_y_array)  *(y - mid_y_array)  )) > inner_radius:
            my_counter += 1
            x = x_coordinates_ALL[my_counter]
            y = y_coordinates_ALL[my_counter]

        for k in range(no_z_electrode,no_z+no_z_electrode):
            for i in range(no_x):
                for j in range(no_y):
                    distance_from_center=sqrt(((i-x)  *(i-x)  )+ ((j-y) *(j-y) ))*dx
                    ion_density=front_factor*exp(-(distance_from_center*distance_from_center)/(b*b))
                    positive_array[i,j,k] += ion_density

                    if positive_array[i,j,k] > MAXVAL:
                        MAXVAL = positive_array[i,j,k]

                    # calculate the recombination only for charge carriers inside the circle
                    if sqrt(((i - mid_x_array)  *(i - mid_x_array)  )+ ((j - mid_y_array) *(j - mid_y_array) )) < inner_radius:
                        no_initialised_charge_carriers+=ion_density

    negative_array=deepcopy(np.asarray(positive_array,dtype='double'))

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
        sx = ion_diff*dt/((dx*dx))
        sy = ion_diff*dt/((dy*dy))
        sz = ion_diff*dt/((dz*dz))

        cx = ion_mobility*Efield*dt/dx*sin(theta)
        cy = 0
        cz = ion_mobility*Efield*dt/dz*cos(theta)

        # check von Neumann's criterion
        von_neumann_expression = (2*(sx + sy + sz) + (cx*cx)+ (cy*cy)+ (cz*cz)<= 1 and (cx*cx)*(cy*cy)*(cz*cz)<= 8*sx*sy*sz)

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

                    # the recombination inside the "inner_radius" circle
                    if sqrt(((i - mid_x_array)   *(i - mid_x_array)   )+ ((j - mid_y_array)  *(j - mid_y_array)  )) < inner_radius:
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