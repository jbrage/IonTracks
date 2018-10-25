# *IonTracks*

Questions? Contact jeppebrage@gmail.com 

## Validation 
The *IonTracks* code is partially validated in    
Christensen, Jeppe Brage, Heikki Tölli, and Niels Bassler. *A general algorithm for calculation of recombination losses in ionization chambers exposed to ion beams* Medical Physics 43.__10__ (2016): 5484-92.

## Calculation of recombination losses
The ```example.py``` file shows how both the initial and total (initial+general) recombination is calculated for a parallel plate ionization chamber exposed to single tracks or a continuous beam

## Dependencies and installation
- The software has been tested on Ubuntu 16.04 and newer
- Following packages are required:

```
sudo apt install python3-numpy python3-scipy python3-matplotlib
sudo apt install mpmath
sudo apt install python3-pip
sudo pip install Cython
```
Use e.g. with
```
cd cython && make
cd .. && python3 example.py
```
