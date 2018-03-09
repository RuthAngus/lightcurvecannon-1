import scipy
from scipy import ndimage 
a = open('training_tgnus_test_unweighted.txt', 'r') 
import pyfits
from scipy import interpolate 
al = a.readlines()

xgrid = arange(2, 270,0.009) 

def interpolate_to_grid(xdata, ydata,xgrid):
  f = interpolate.interp1d(xdata, ydata)
  new_ydata= f(xgrid)
  return xgrid, new_ydata

al2 = [] 
for each in al:
    al2.append(each.split()[0]) 
freqall,fluxall = [],[] 
counter = 0
for each in al2:
    print counter
    b = pyfits.open(each) 
    freq = b[1].data['FREQUENCY'] 
    flux = b[1].data['PSD'] 
    newx,newy = interpolate_to_grid(freq,flux,xgrid)
    freqall.append(newx)#[0:5000])
    fluxall.append(newy)#[0:5000]) 
    counter = counter+1

ifluxall = [] 
#for each in fluxall[0:10000]:
for each in fluxall[0:2000]: # used to use the full 10,000 but now only use the first 2000 
    ifluxall.append(fft.ifft(each)) 

import numpy as np
import pyfits
import pickle
import matplotlib.pyplot as plt
 
tc_fluxa = abs(real(ifluxall))[:,0:2000]
tc_fluxb = abs(real(ifluxall))[:,-2000:]
test = (arctan(imag(ifluxall)/real(ifluxall)))
dtest = abs(diff(test))
tc_fluxa_log = log(abs(real(ifluxall)))[:,0:27500] # if log the trend is flat but the scatter is far larger 
tc_fluxa = (abs(real(ifluxall)))[:,0:27500] # if log the trend is flat but the scatter is far larger 
tc_flux = tc_fluxa_log
tc_flux_linear = tc_fluxa
tc_wavelx = [] 
tc_error = []
for each in tc_flux_linear:
    tc_error.append(1./each**0.5) # this gives best performance
    tc_wavelx.append(arange(0, len(each), 1))

error_take = array(tc_error) 
bad = isinf(error_take) 
labels = ['teff', 'logg', 'numax', 'deltanu','c', 'n']
nmeta = len(labels) 
teff, logg, numax, deltanu = loadtxt('training_tgnus.txt', usecols = (1,2,3,4), unpack =1) 
c,n,feh = loadtxt('training_tgnuscn.txt', usecols = (5,6,7), unpack =1) 
teff_train = teff[0:500]
logg_Train = logg[0:500]
logg_seismic = loadtxt("training_tgnuscn_seismic.txt", usecols = (2,), unpack =1) 
logg_train_seismic, logg_test_seismic = logg_seismic[0:5000], logg_seismic
teff_test = teff
logg_test  = logg
deltanu_test, numax_test = deltanu, numax
c_train, n_train = c[0:500], n[0:500]
c_test, n_test = c,n 
feh_train, feh_test = feh[0:500], feh


tc_names = al2
tc_names_train = al2[0:500] 
tc_names_test = al2
metaall = np.ones((len(tc_names), nmeta))
countit = np.arange(0,len(tc_flux),1)
newwl = np.arange(0,len(tc_flux),1) 
npix = np.shape(tc_flux[0]) [0]
 
dataall = np.zeros((npix, len(tc_names), 3))
for a,b,c,jj in zip(tc_wavelx, tc_flux, tc_error, countit):
    dataall[:,jj,0] = a
    dataall[:,jj,1] = b
    dataall[:,jj,2] = c

nstars = np.shape(dataall)[1]
for k in range(0,len(tc_names)):
    metaall[k,0] = teff_test[k]
    metaall[k,1] = logg_test[k]
    metaall[k,2] = numax_test[k]
    metaall[k,3] = deltanu_test[k]
    metaall[k,4] = c_rand[k] # c_test[k]
    metaall[k,5] = n_rand[k] # n_test[k]
 
a = open('listall_tgnus_test.txt', 'r') 
al = a.readlines()
al2 = []
for each in al:
    al2.append(each.strip())
kepids_a = []
for each in al:
    kepids_a.append(each.split('/')[2].split('.fits')[0].split('kplr')[1].split('_')[0]) 
kepids = []
for each in kepids_a:
    if logical_and(each[0] == '0', each[1] != '0'):
        kepids.append(each[1:])
    elif logical_and(each[0] == '0', each[1] == '0'):
        kepids.append(each[2:])


file_in = open('test_realifft_unweighted.pickle', 'wb') 
pickle.dump((dataall[:,0:2000,:], metaall[0:2000,:], labels, tc_names_test[0:2000], tc_names_test[0:2000]),  file_in)
file_in.close()
