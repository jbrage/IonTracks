# _IonTracks_

Questions? Contact jeppe.christensen@psi.ch 

## _IonTracks_ calculations

Features the calculation of recombination losses in ion, electron, and photon beams in parallel-plate gas-filled ionization chambers:

- Ion beams exhibit a non-uniform charge carrier distribution which is accounted for through amorphous track structure theory.
  Use the "IonBeams" scripts for ions.
- Photon and electron beams are associated with uniform charge carrier densities. The current scripts enable the inclusion of electric fields varying in both time and space in constrast to theories.

## Documentation and validation

Details can be found under ```documentation``` including a note about the differences between simulating recombination in photon/electron or ion beams. 


#### Validation of recombination in light ion beams:

The _IonTracks_ code is partially validated in
Christensen, Jeppe Brage, Heikki Tölli, and Niels Bassler (2016) A general algorithm for calculation of recombination losses in ionization chambers exposed to ion beams _Medical Physics_ 43.**10**: 5484-92 https://aapm.onlinelibrary.wiley.com/doi/full/10.1118/1.4962483

and
Christensen, Jeppe Brage _et al_ (2020) Mapping initial and general recombination in scanning proton pencil beams _Phys. Med. Biol._ **65** 115003
https://iopscience.iop.org/article/10.1088/1361-6560/ab8579

#### Validation of recombination in electron and photon beams:

_IonTracks_ is validated against the Boag theory (the basis of the Two-voltage method) for photon and electron beams with uniform charge carrier densities, see ```documentation```.

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
source venv/bin/activate
```

There are two ways to compile the package with requirements:

* Compile this package and all its basic requirements:

  ```
  CYTHONIZE=1 pip install --editable .
   ```

* If you have a GPU and want to compute using the cupy library, compile the package with the base requirements, along with an additional GPU requirement:
  * For the cuda 11x version:
      ```
      CYTHONIZE=1 pip install --editable ".[gpu-cuda11x]"
      ```
  * For the cuda 12x version:
      ```
      CYTHONIZE=1 pip install --editable ".[gpu-cuda12x]"
      ```

Run the example script:

```
python hadrons/example_single_track.py
```

When you are done, deactivate your virtual enviroment:

```
deactivate
```
