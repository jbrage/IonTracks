# *IonTracks*

Questions? Contact jeppebrage@gmail.com (mail addresses given in the papers are obsolete)

## *IonTracks* calculations
Features the calculation of recombination losses in ion, electron, and photon beams. 
Ion beams exhibit a non-uniform charge carrier distribution which is accounted for through amorphous track structure theory.
Use the "IonBeams" scripts for ions.

Photon and electron beams are associated with uniform charge carrier densities. The current scripts enable the inclusion of electric fields varying in both time and space in constrast to theories.


## Validation 
#### Ion beams:
The *IonTracks* code is partially validated in    
Christensen, Jeppe Brage, Heikki Tölli, and Niels Bassler. *A general algorithm for calculation of recombination losses in ionization chambers exposed to ion beams* Medical Physics 43.__10__ (2016): 5484-92.

#### Photon and electron beams:
*IonTracks* is validated against the Boag theory (the basis of the Two-voltage method) for photon and electron beams with uniform charge carrier densities 

## Calculation of recombination losses
The ```example.py``` file shows how both the initial and total (initial+general) recombination is calculated for a parallel plate ionization chamber exposed to single tracks or a continuous beam

## Dependencies and installation
- The software has been tested on Ubuntu 16.04 and newer
- Following packages are required:

```
sudo apt install python3-numpy python3-scipy python3-matplotlib python3-mpmath
sudo apt install python3-pip
sudo pip install Cython
```
Use e.g. with
```
cd cython && make
cd .. && python3 example.py
```
