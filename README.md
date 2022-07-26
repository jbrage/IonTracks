# _IonTracks_

Questions? Contact jeppebrage@gmail.com (mail addresses given in the papers are obsolete)

## _IonTracks_ calculations

Features the calculation of recombination losses in ion, electron, and photon beams in parallel-plate gas-filled ionization chambers:

- Ion beams exhibit a non-uniform charge carrier distribution which is accounted for through amorphous track structure theory.
  Use the "IonBeams" scripts for ions.
- Photon and electron beams are associated with uniform charge carrier densities. The current scripts enable the inclusion of electric fields varying in both time and space in constrast to theories.

## Validation

#### Ion beams:

The _IonTracks_ code is partially validated in
Christensen, Jeppe Brage, Heikki Tölli, and Niels Bassler (2016) A general algorithm for calculation of recombination losses in ionization chambers exposed to ion beams _Medical Physics_ 43.**10**: 5484-92 https://aapm.onlinelibrary.wiley.com/doi/full/10.1118/1.4962483

and
Christensen, Jeppe Brage _et al_ (2020) Mapping initial and general recombination in scanning proton pencil beams _Phys. Med. Biol._ **65** 115003
https://iopscience.iop.org/article/10.1088/1361-6560/ab8579

#### Photon and electron beams:

_IonTracks_ is validated against the Boag theory (the basis of the Two-voltage method) for photon and electron beams with uniform charge carrier densities

## Calculation of recombination losses

Both the IonBeams folder and the electron/photon folder feature example files.

## Dependencies and installation

- The software has been tested on Ubuntu 20.04
- Following packages are required:

```
sudo apt install python3-pip python3-venv
```

Create python virtual environment `venv`, a directory which will hold compiled version of this package and all the libraries it needs to run (more on venv: https://docs.python.org/3/library/venv.html):

```
python -m venv venv
```

Compile this package and all its requirements:

```
CYTHONIZE=1 pip install --editable .
```

Run the example script:

```
python hadrons/example_single_track.py
```

When you are done, deactivate your virtual enviroment:

```
deactivate
```
