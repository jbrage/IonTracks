# IonTracks

Christensen, Jeppe Brage, Heikki Tölli, and Niels Bassler. "A general algorithm for calculation of recombination losses in ionization chambers exposed to ion beams." Medical Physics 43.10 (2016): 5484-5492.

The code IonTracks consists of different parts, see below. The idea is to distribute charge carriers in ion tracks depending on the linear energy transfer (LET) and a Gaussian track radius (b).

## Comparison with the Jaffé theory and initial (intra-track) recombination in a single track:
- main.py compares IonTracks with the Jaffé theory for initial recombination.

## Recombination in a pulsed beam:
- main_pulsed.py distributes particle tracks in an array and computes the total initial and general recombination in a pulsed beam.
- it is assumed, that the ionization of the medium between the electrodes is instantaneous.
- the sample_from_spectra files are an extension to the pulsed_beam files
- the simulation can be improved by generating spectra (.spc) files where the scripts SH_spc_files.py and get_b_LET.py sample from the energy spectra and print corresponding LET and Gaussian track radii to a file

## Dependencies and installation
- Generally, the simulation of charge carriers is computed with python3 and is written on Ubuntu 16.04:

```
sudo apt install python3 
sudo apt install python3-numpy python3-scipy python3-matplotlib
sudo apt install python-pip
sudo apt install mpmath
sudo pip install Cython
```
- Cython version 0.23 or newer is required
```
make && "python3 main.py
```

- the sampling of particle parameters from .spc files requires installion of libamtrack:
    - https://github.com/libamtrack/library
    - the created libamtrack.so file must be compiled locally

- an example of .spc files can be downloaded here: 
    - https://libamtrack.dkfz.org/libamtrack/index.php/Download

- .spc files for a particular beam setup can be created with e.g. SHIELD-HIT:
    - http://shieldhit.org/ 





