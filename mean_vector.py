import os
import numpy as np
import pathlib
import pandas as pd
import pickle
from pickle import dump
a=[]
pathfile=os.listdir('C:/Users/DELL/Desktop/ramsi_pkl')
#print (pathfile)
paTh='C:/Users/DELL/Desktop/ramsi_pkl'
for filename in pathfile:
	#os.path.abspath(paTh)
	path=paTh +"/"+filename
	x= pickle.load(open(path,'rb'))
	#print(filename)
	a.append(x)
#print(a)
ramsi_avg=np.mean(a,axis=0)
print(ramsi_avg)
dump(ramsi_avg, open('C:/Users/DELL/Desktop/ramsivgg16_mean.pkl', 'wb'))
