# *IonTracks*

Questions? Contact jeppebrage@gmail.com (mail addresses given in the papers are obsolete)

## *IonTracks* calculations
Features the calculation of recombination losses in ion, electron, and photon beams in parallel-plate gas-filled ionization chambers: 
- Ion beams exhibit a non-uniform charge carrier distribution which is accounted for through amorphous track structure theory.
Use the "IonBeams" scripts for ions.

- Photon and electron beams are associated with uniform charge carrier densities. The current scripts enable the inclusion of electric fields varying in both time and space in constrast to theories.


## Validation 
#### Ion beams:
The *IonTracks* code is partially validated in    
Christensen, Jeppe Brage, Heikki Tölli, and Niels Bassler (2016) A general algorithm for calculation of recombination losses in ionization chambers exposed to ion beams *Medical Physics* 43.__10__: 5484-92  
https://aapm.onlinelibrary.wiley.com/doi/full/10.1118/1.4962483

and 
Christensen, Jeppe Brage *et al* (2020) Mapping initial and general recombination in scanning proton pencil beams *Phys. Med. Biol.* __65__ 115003
https://iopscience.iop.org/article/10.1088/1361-6560/ab8579

#### Photon and electron beams:
*IonTracks* is validated against the Boag theory (the basis of the Two-voltage method) for photon and electron beams with uniform charge carrier densities 

## Calculation of recombination losses
Both the IonBeams folder and the electron/photon folder feature example files.


## Dependencies and installation
- The software has been tested on Ubuntu 16.04 and newer
- Following packages are required:

```
sudo apt install python3-numpy python3-scipy python3-matplotlib python3-mpmath
sudo apt install python3-pip
sudo pip install Cython
```
Run an example of the IonBeams with
```
cd IonBeams/cython && make
cd ../ && python3 example.py
```
