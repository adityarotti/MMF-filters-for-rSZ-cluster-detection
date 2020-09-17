##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester
# Date created: 20 March 2020
# Date modified: 20 August 2020
##################################################################################################
import pickle

def write_dict(filename,dict):
	with open(filename, 'wb') as handle:
		pickle.dump(dict, handle)

def read_dict(filename):
	with open(filename, 'rb') as handle:
		dict = pickle.loads(handle.read())
	return dict


