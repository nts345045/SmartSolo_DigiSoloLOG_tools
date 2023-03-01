"""
:module: estimate_sta_locs.py
:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu
:purpose: Take parsed DigiSolo.LOG data and use it to estimate station locations with bivariate statistical estimators

scikit-learn implementation of DBSCAN from: https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
"""

import os
import pandas as pd
import numpy as np
from glob import glob
from pyproj import Proj
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Define WSG84 to UTM12 coordinate transform
myproj = Proj('epsg:32712')


def load_parsed_LOGcsv(LOGcsv_fname):
	df = pd.read_csv(LOGcsv_fname,parse_dates=True,index_col=[0])
	return df

def run_loc_estimate(df_LOG,tstart=pd.Timestamp('1970-01-01'),tend=pd.Timestamp('2024-01-01'),coord_convert=myproj,sta_name=None,src=None):
	"""
	Calculate a range of statistical estimates of station locations for a specified acquisition period
	:: INPUTS ::
	:param df_LOG: pandas.DataFrame of loaded Parsed_DigiSolo_LOG.csv data with a pandas.DatetimeIndex index
	:param tstart: minimum timestamp for data in LOGcsv to consider (default is 1970-01-01)
	:param tend: maximum timestamp for data in LOGcsv to consider (default is 2024-01-01)
	:param coord_convert: pyproj.Proj defined method for converting from Lat/Lon into a UTM CRS
	:param sta_name: name for output series

	:: OUTPUT ::
	:return S_loc: pandas.Series with coordinate means, covariance matrix entries (upper triangle), sample counts, median
	:return df_UTM: pandas.DataFrame with outputs of converting LLH --> ENH (H = Altitude or Height above sea-level in meters)
	"""
	# Filter out NaN values
	IND = (df_LOG['Latitude'].notna()) & (df_LOG['Longitude'].notna())
	# Convert into cartesian
	mE,mN = coord_convert(df_LOG[IND]['Longitude'].values,df_LOG[IND]['Latitude'].values)
	mH = df_LOG[IND]['Altitude'].values
	# Calculate Meter-Scaled Covariance Matrix
	covENZ = np.cov(np.array([mE,mN,mH]))
	# Calculate & compile estimates
	d_loc = {'mE mean':np.nanmean(mE),'mN mean':np.nanmean(mN),'mH mean':np.nanmean(mH),'lat mean':df_LOG['Latitude'].mean(),'lon mean':df_LOG['Longitude'].mean(),\
			 'mE var':covENZ[0,0],'mN var':covENZ[1,1],'mH var':covENZ[2,2],'mEmN cov':covENZ[0,1],'mEmH cov':covENZ[0,2],'mNZ cov':covENZ[1,2],\
			 'mE med':np.nanmedian(mE),'mN med':np.nanmedian(mN),'mH med':np.nanmedian(mH),'npts':len(mE),'src':src,\
			 'tstart':df_LOG[IND].index.min(),'tend':df_LOG[IND].index.max(),'KMeans':False,'Az mean':df_LOG[IND]['eCompass North'].mean(),\
			 'Az var':df_LOG[IND]['eCompass North'].std()**2}
	# Form Series and DataFrame for output
	S_out = pd.Series(data=d_loc,index=d_loc.keys(),name=sta_name)
	df_UTM = pd.DataFrame({'mE':mE,'mN':mN,'mH':mH,'Az':df_LOG[IND]['eCompass North'].values},index=df_LOG[IND].index)
	return S_out, df_UTM


def run_DBSCAN_4D(df_ENH,eps=0.3,min_samp=10):
	# Make Elapsed Time Vector
	sDT = (df_ENH.index - df_ENH.index[0]).total_seconds().values
	X = np.array([df_ENH['mE'].values,df_ENH['mN'].values,df_ENH['mH'].values,sDT]).T
	X = StandardScaler().fit_transform(X)
	db = DBSCAN(eps=eps,min_samples=min_samp).fit(X)
	return X,db

def run_KMeans_4D(df_ENH,n_clusters=2,n_init=10):
	# Make Elapsed Time Vector
	sDT = (df_ENH.index - df_ENH.index[0]).total_seconds().values
	X = np.array([df_ENH['mE'].values,df_ENH['mN'].values,df_ENH['mH'].values,sDT]).T
	X = StandardScaler().fit_transform(X)
	y_pred = KMeans(n_clusters=n_clusters,n_init=n_init).fit_predict(X)
	df_out = pd.concat([df_ENH.copy(),pd.Series(y_pred,index=df_ENH.index,name='Cluster #')],axis=1,ignore_index=False)
	return df_out



ROOT = os.path.join('..','..','processed_data','passive','metadata','LOG')
DATA = glob(os.path.join(ROOT,'*','*','Parsed_DigiSolo_LOG.csv'))

name_dir = {'453001556':{'sta_name':'7100','tstart':pd.Timestamp('2023-01-06')},\
			'453010437':{'sta_name':'7101','tstart':pd.Timestamp('2023-01-06')},\
			'453001497':{'sta_name':'7102','tstart':pd.Timestamp('2023-01-06')},\
			'453007395':{'sta_name':'7103','tstart':pd.Timestamp('2023-01-06')},\
			'453007390':{'sta_name':'7104','tstart':pd.Timestamp('2023-01-06')},\
			'453005412':{'sta_name':'7105','tstart':pd.Timestamp('2023-01-06')},\
			'453007451':{'sta_name':'7309','tstart':pd.Timestamp('2023-01-10')},\
			'453005629':{'sta_name':'7310','tstart':pd.Timestamp('2023-01-10')},\
			'453009814':{'sta_name':'7311','tstart':pd.Timestamp('2023-01-10')},\
			'453010673':{'sta_name':'7312','tstart':pd.Timestamp('2023-01-10')},\
			'453007380':{'sta_name':'DASN1','tstart':pd.Timestamp('2023-01-10')},\
			'453001303':{'sta_name':'DASN2'}}

# Names in chronological order

multi_deploy = {'7101':(2,'WSD0','7101'),\
				'7311':(2,'WSD1','7311'),\
				'7312':(2,'WSD2','7312'),\
				'DASN2':(2,'DASN2','DASN3')}


import matplotlib.pyplot as plt
df_means = pd.DataFrame()
plt.figure()
for f_ in DATA:
	print('=== RUNNING: %s ==='%(f_))
	FPATH,FILE = os.path.split(f_)
	iPATH,ONTS = os.path.split(FPATH)
	jPATH,DAS = os.path.split(iPATH)
	if DAS in name_dir.keys():
		# Load data
		_df_ = load_parsed_LOGcsv(f_)
		# Run initial estimation of mean position
		iS_out, idf_out = run_loc_estimate(_df_,**name_dir[DAS],src=f_)
		# If flagged as a multi-deploy, run KMeans clustering
		if name_dir[DAS]['sta_name'] in list(multi_deploy.keys()):
			# Pull number of expected clusters
			n_clust = multi_deploy[name_dir[DAS]['sta_name']][0]
			# Run clustering
			idf_out = run_KMeans_4D(idf_out,n_clusters=n_clust)
			# Iterate across cluster ID #'s and re-run averaging
			for j_ in idf_out['Cluster #'].unique():
				# Get sub-subset data
				jdf_out = idf_out[idf_out['Cluster #']==j_]
				# Create master index
				JND = _df_.index.isin(jdf_out.index)
				# Get jname
				if jdf_out.index.min() > idf_out.index.min():
					jname = multi_deploy[name_dir[DAS]['sta_name']][2]
				else:
					jname = multi_deploy[name_dir[DAS]['sta_name']][1]
				# Run new covariance matrix calc
				# breakpoint()
				covENZ = np.cov(jdf_out[['mE','mN','mH']].values.T)
				# Compile new estimates
				d_loc = {'mE mean':jdf_out['mE'].mean(),'mN mean':jdf_out['mN'].mean(),'mH mean':jdf_out['mH'].mean(),'Az mean':np.nanmean(jdf_out['Az'].values),\
						 'lat mean':_df_[JND]['Latitude'].mean(),'lon mean':_df_[JND]['Longitude'].mean(),\
			 			 'mE var':covENZ[0,0],'mN var':covENZ[1,1],'mH var':covENZ[2,2],'Az var':np.nanstd(jdf_out['Az'].values)**2,\
			 			 'mEmN cov':covENZ[0,1],'mEmH cov':covENZ[0,2],'mNZ cov':covENZ[1,2],\
						 'mE med':jdf_out['mE'].median(),'mN med':jdf_out['mN'].median(),'mH med':jdf_out['mH'].median(),\
						 'npts':len(jdf_out),'src':f_,'KMeans':True,'tstart':_df_[JND].index.min(),'tend':_df_[JND].index.max()}

				# Form Series and DataFrame for output
				iS_out = pd.Series(data=d_loc,index=d_loc.keys(),name=jname)
				df_means = pd.concat([df_means,iS_out],axis=1,ignore_index=False)
				# Write clustered coordinates to 
				jdf_out.to_csv(os.path.join(FPATH,'ENH_%s_epsg32712.csv'%(jname)),index=True,header=True)
				plt.plot(jdf_out['mE'],jdf_out['mN'],'.',alpha=0.1)
				plt.plot(iS_out['mE mean'],iS_out['mN mean'],'r*')
				plt.text(iS_out['mE mean'],iS_out['mN mean'],jname)
		else:
			idf_out = pd.concat([idf_out,pd.Series(data=np.zeros(len(idf_out)),index=idf_out.index,name='Cluster #')],axis=1,ignore_index=False)
			jname = name_dir[DAS]['sta_name']
			idf_out.to_csv(os.path.join(FPATH,'ENH_%s_epsg32712.csv'%(jname)),header=True,index=True)	
			df_means = pd.concat([df_means,iS_out],axis=1,ignore_index=False)

			plt.plot(idf_out['mE'],idf_out['mN'],'.',alpha=0.1)
			plt.plot(iS_out['mE mean'],iS_out['mN mean'],'r*')
			plt.quiver(iS_out['mE mean'],iS_out['mN mean'],np.cos((np.pi/180.)*(90. - iS_out['Az mean'])),\
														   np.sin((np.pi/180.)*(90. - iS_out['Az mean'])))
			plt.text(iS_out['mE mean'],iS_out['mN mean'],jname)
	else:
		_df_ = load_parsed_LOGcsv(f_)

		breakpoint()

plt.show()
# Save to disk - some duplicate stations may exist from partial duplicate log files - additional QC needed by analyst.
df_means.T.sort_index().to_csv(os.path.join(ROOT,'KMeans_Clustered_Estimates.csv'),header=True,index=True)

