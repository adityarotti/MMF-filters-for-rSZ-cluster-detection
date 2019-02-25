from settings import mmf_settings as mmfset
from data_preprocess import preprocess_pico_sims as pps
mmfset.init()
#pps.get_reduced_pico_sims()
pps.extract_tangent_planes(numplanes=10,dryrun=False)
