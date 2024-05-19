from __future__ import division
from abc import ABC


class GenericElectronSolver(ABC):
    def __init__(
        self,
        electron_density_per_cm3,  # fluence-rate [/cm^2/s]
        voltage_V,  # [V/cm] magnitude of the electric field
        electrode_gap,  # [cm] # electrode gap
        unit_length_cm,
        ion_mobility=1.73,  # cm s^-1 V^-1, averaged for positive and negative ions
        ion_diff=3.7e-2,  # cm^2/s, averaged for positive and negative ions
        alpha=1.60e-6,  # cm^3/s, recombination constant
        r_cm=0.002,  # radius of the sampled cylinder
        buffer_radius=5,
    ):
        """
        Define the grid size parameters
        """
        # LET_eV_cm = LET_keV_um*1e7
        no_xy = int(2 * r_cm / unit_length_cm)  # no of voxels in xy-directions
        no_xy += 2 * buffer_radius
        no_z = int(
            electrode_gap / unit_length_cm
        )  # number of elements in the z direction
        no_z_electrode = 5  # length of the electrode-buffer to ensure no ions drift through the array in one time step
        no_z_with_buffer = 2 * no_z_electrode + no_z
        # depending on the cluster/computer, the upper limit may be changed
        if (no_xy * no_xy * no_z) > 1e8:
            raise ValueError(
                "Too many elements in the array: %i"
                % (no_xy * no_xy * no_z_with_buffer)
            )
        """
        find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
        decreases and eventually damps out
        """
        dt = 1.0
        von_neumann_expression = False
        Efield_V_cm = (
            voltage_V / electrode_gap
        )  # to ensure the same dt in all simulations
        while not von_neumann_expression:
            dt /= 1.01
            # as defined in the Deghan (2004) paper
            # sx = ion_diff*dt/(self.unit_length_cm**2)
            # sy = ion_diff*dt/(self.unit_length_cm**2)
            sx = sy = (
                0.0  # uniform charge => no gradient driven diffusion in the xy plane
            )

            sz = ion_diff * dt / (unit_length_cm**2)
            cx = cy = 0.0
            cz = ion_mobility * Efield_V_cm * dt / unit_length_cm
            # check von Neumann's criterion
            von_neumann_expression = (
                2 * (sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1
                and cx**2 * cy**2 * cz**2 <= 8 * sx * sy * sz
            )
        """
        Calculate the number of step required to drag the two charge carrier distributions apart
        along with the number of tracks to be uniformly distributed over the domain
        """
        separation_time_steps = int(
            electrode_gap / (2.0 * ion_mobility * Efield_V_cm * dt)
        )
        self.computation_time_steps = separation_time_steps * 3
        self.no_xy = no_xy
        self.no_z_with_buffer = no_z_with_buffer
        self.no_z_electrode = no_z_electrode
        self.dt = dt
        self.no_z = no_z
        self.electron_density_per_cm3 = electron_density_per_cm3
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.alpha = alpha
