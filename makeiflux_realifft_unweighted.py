import scipy
from scipy import ndimage 
a = open('training_tgnus_test_unweighted.txt', 'r') 
import pyfits
from scipy import interpolate 
numtrain = 500
al = a.readlines()[0:numtrain]

xgrid = arange(2, 270,0.009) 
diff_freq = diff(xgrid)
Per = 1/diff_freq/(10**-6)
diff_time = Per/len(xgrid)


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
    freqall.append(newx)#[0:5000])#[0:5000])
    fluxall.append(newy)#[0:5000])#[0:5000]) 
    counter = counter+1

ifluxall = [] 
for each in fluxall: # use the first 2000 spectra 
    #each = each[0:1000]
    ifluxall.append(fft.ifft(each)) 

import numpy as np
import pyfits
import pickle
import matplotlib.pyplot as plt
 
tc_fluxa = abs(real(ifluxall))[:,0:2000]
tc_fluxb = abs(real(ifluxall))[:,-2000:]
test = (arctan(imag(ifluxall)/real(ifluxall)))
# 2000 is the best number so far 
tc_fluxa_log = log(abs(real(ifluxall)))[:,0:27500] # if log the trend is flat but the scatter is far larger 
tc_fluxa = abs(real(ifluxall))[:,0:27500] # if log the trend is flat but the scatter is far larger 
tc_flux = tc_fluxa_log
tc_flux_linear = tc_fluxa
tc_wavelx = [] 
tc_error = []
for each in tc_flux_linear:
    tc_error.append(1./each**0.5) # this gives best performance
    #tc_error.append(1./each**1.0) # this gives worse performance
    tc_wavelx.append(arange(0, len(each), 1))

error_take = array(tc_error) 
bad = isinf(error_take) 
labels = ['teff', 'logg', 'numax', 'deltanu', 'c', 'n']
nmeta = len(labels) 
teff,logg,feh,alpha,mass = loadtxt('training_realifft.txt', usecols = (1,2,3,4,5), unpack =1) 
teff,logg, numax, deltanu = loadtxt("training_tgnus.txt", usecols = (1,2,3,4), unpack =1) 
c_feh,n_feh,feh = loadtxt("training_tgnuscn.txt", usecols = (5,6,7), unpack =1) 
teff_train = teff[0:numtrain]
logg_train, feh_train, alpha_train, mass_train = logg[0:numtrain], feh[0:numtrain], alpha[0:numtrain], mass[0:numtrain] 
numax_train = numax[0:numtrain]
deltanu_train = deltanu[0:numtrain] 
c_train,n_train = c_feh[0:numtrain], n_feh[0:numtrain]
c_test,n_test = c_feh, n_feh
feh_train,feh_test = feh[0:numtrain], feh
logg_seismic = loadtxt("training_tgnuscn_seismic.txt", usecols = (2,), unpack =1) 
logg_train_seismic, logg_test_seismic = logg_seismic[0:numtrain], logg_seismic

kepids = []
a = open('kepids_4664.txt', 'r')
al = a.readlines()
for each in al:
    kepids.append(each.strip())

kepids = array(kepids) 
tc_names = kepids[0:numtrain] 
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
for k in range(0,len(feh_train)):
    metaall[k,0] = teff_train[k]
    metaall[k,1] = logg_train[k]
    metaall[k,2] = numax_train[k]
    metaall[k,3] = deltanu_train[k]
    metaall[k,4] = c_rand[k] #c_train[k]
    metaall[k,5] = n_rand[k] #n_train[k]
 
a = open('listall_tgnus.txt', 'r') 
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

file_in = open('training_realifft_unweighted.pickle', 'w') 
pickle.dump((dataall, metaall, labels, tc_names, al2),  file_in)
file_in.close()
