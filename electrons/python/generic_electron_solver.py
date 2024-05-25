from __future__ import division
from abc import ABC
from dataclasses import dataclass


@dataclass
class GenericElectronSolver(ABC):
    # Simulation parameters
    electron_density_per_cm3: float  # fluence-rate [/cm^2/s]
    voltage_V: float  # [V/cm] magnitude of the electric field
    electrode_gap: float  # [cm] # electrode gap
    unit_length_cm: float
    ion_mobility: float = 1.73  # cm s^-1 V^-1, averaged for positive and negative ions
    ion_diff: float = 3.7e-2  # cm^2/s, averaged for positive and negative ions
    alpha: float = 1.60e-6  # cm^3/s, recombination constant
    r_cm: float = 0.002  # radius of the sampled cylinder
    buffer_radius: int = 5
    no_z_electrode: int = (
        5  # length of the electrode-buffer to ensure no ions drift through the array in one time step
    )
    dt: float = 1.0

    # Derived properties
    @property
    def no_xy(self) -> int:
        """Number of voxels om the xy-directions"""

        no_xy = int(2 * self.r_cm / self.unit_length_cm)
        no_xy += 2 * self.buffer_radius
        return no_xy

    @property
    def no_z(self) -> int:
        """Number of voxels om the z-direction"""

        return int(self.electrode_gap / self.unit_length_cm)

    @property
    def no_z_with_buffer(self) -> int:
        return 2 * self.no_z_electrode + self.no_z

    @property
    def Efield_V_cm(self) -> float:
        return self.voltage_V / self.electrode_gap

    @property
    def separation_time_steps(self) -> int:
        """
        Number of step required to drag the two charge carrier distributions apart
        along with the number of tracks to be uniformly distributed over the domain
        """
        return int(
            self.electrode_gap / (2.0 * self.ion_mobility * self.Efield_V_cm * self.dt)
        )

    @property
    def computation_time_steps(self) -> int:
        return self.separation_time_steps * 3

    def von_neumann_expression(self) -> None:
        """
        Finds a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
        decreases and eventually damps out
        """
        von_neumann_expression = False

        while not von_neumann_expression:
            self.dt /= 1.01
            # as defined in the Deghan (2004) paper
            # sx = ion_diff*dt/(self.unit_length_cm**2)
            # sy = ion_diff*dt/(self.unit_length_cm**2)
            self.sx = self.sy = (
                0.0  # uniform charge => no gradient driven diffusion in the xy plane
            )

            self.sz = self.ion_diff * self.dt / (self.unit_length_cm**2)
            self.cx = self.cy = 0.0
            self.cz = (
                self.ion_mobility * self.Efield_V_cm * self.dt / self.unit_length_cm
            )
            # check von Neumann's criterion
            criterion_1 = (
                2 * (self.sx + self.sy + self.sz) + self.cx**2 + self.cy**2 + self.cz**2
                <= 1
            )

            criterion_2 = (
                self.cx**2 * self.cy**2 * self.cz**2 <= 8 * self.sx * self.sy * self.sz
            )

            von_neumann_expression = criterion_1 and criterion_2

    def __post_init__(
        self,
    ):
        # depending on the cluster/computer, the upper limit may be changed
        if (self.no_xy * self.no_xy * self.no_z) > 1e8:
            raise ValueError(
                "Too many elements in the array: %i"
                % (self.no_xy * self.no_xy * self.no_z_with_buffer)
            )

        self.von_neumann_expression()
