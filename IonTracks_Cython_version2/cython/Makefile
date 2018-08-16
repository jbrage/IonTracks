all:
	python3 setup.py build_ext --inplace
	python3 /home/jepb/.local/lib/python3.5/site-packages/cython.py -a distribute_and_evolve_pulsed.pyx

clean:
	rm *.c *.so *.html
	rm -r build
