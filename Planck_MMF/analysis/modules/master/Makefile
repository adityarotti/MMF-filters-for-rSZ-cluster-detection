# Path to the moduales
subroutines=./subroutines

#LDFLAGS	= -llapack -lblas -lm -lc
LDFLAGS	= liblapack.so libblas.so -lm -lc

FC=gfortran -fbounds-check -march=native -cpp -ffixed-line-length-none -fPIC #-Wall -Wextra -Wconversion

all:master
master:master_obj
	f2py -c --fcompiler=gfortran $(LDFLAGS) *.o -m bmaster master.f90
	make clean

master_obj:
	$(FC) -c $(F90flags) $(subroutines)/wigner/*.f 	

clean:
	$(RM) -r *.o *.mod *~ *dSYM *.pyc

cleanall:
	$(RM) *.o *.mod *.so *.pyf *~
