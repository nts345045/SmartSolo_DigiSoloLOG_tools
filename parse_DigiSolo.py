"""
:module parse_DigiSolo.py:
:auth: Nathan T. Stevens
:email: nts5045@psu.edu | ntstevens@wisc.edu
:purpose: Methods for converting SmartSolo DigiSolo.LOG files into time-indexed, tabuldated data

Basic Use: df = parse_blocks('DigiSolo.LOG')
Produces a pandas.DataFrame with a DatetimeIndex from target DigiSolo.LOG in the current working directory

:LAST UPDATED: 22. Feb. 2023
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob


### SUBROUTINES ###

def getlines(ifile):
	"""
	Solution from: https://stackoverflow.com/questions/25047254/parse-blocks-of-text-from-text-file-using-python
	"""
	outs = open(ifile).read().split('\n')
	return outs

def getblocklength(lines,i_start,flag=''):
	"""
	Get the number of line entries from a subset of lines

	:: INPUTS ::
	:type lines: list
	:param lines: list of text lines
	:type i_start: int
	:param i_start: index to start counting from
	:type flag: str
	:param flag: string that signals end of block. Default is an empty line: ''

	:: OUTPUT ::
	:rtype j_: int
	:return j_: length of queried block
	"""
	j_ = 0
	done = False
	while not done:
		il = lines[i_start + j_]
		if il == flag:
			done = True
		else:
			done = False
			j_ += 1
	return j_


def parse_block(iblock):
	"""
	Subroutine for parsing data within identified blocks, returns a series
	"""
	idict = {'Block':iblock[0]}
	for i_ in np.arange(1,len(iblock)):
		# try:
		if '=' in iblock[i_]:
			key,val = iblock[i_].split('=')
			key = re.sub('\s+',' ',key)[:-1]
			if 'Time' in key:
				tval = val.split('"')[1].split(',')
				dtval = pd.Timestamp('%sT%s'%(tval[0],tval[1]))
				if 'UTC Time' == key:
					idx = dtval
				else:
					idict.update({key:dtval})
			elif key not in ['GPS Strength','Satellite ID']:
				try:
					idict.update({key:float(val)})
				except ValueError:
					idict.update({key:val})
			elif key == 'GPS Strength':
				GPS_array = np.array([val.split('"')[1].split(',')],dtype=int)
				idict.update({'Satellite Strength Mean':np.mean(GPS_array)})
		# except:
		# 	breakpoint()
	S_ = pd.Series(idict,name=idx)
	return S_


#### CORE PROCESS ####

def getblocks(ifile,start_delim='[',stop_flag=''):
	blocks = []
	df_ = pd.DataFrame()
	i_ = 0
	lines = getlines(ifile)
	for i_ in tqdm(range(len(lines))):
	# while i_ < len(lines):
		l_ = lines[i_]
		blocktop = False
		if start_delim in l_:
			bl_ = getblocklength(lines,i_,flag=stop_flag)
			iblock = lines[i_:i_+bl_]
			if 'DeviceInfo' in iblock[0]:
				print('Skipping [DeviceInfo*] block starting at line %d'%(i_+1))
			elif len(iblock) < 2:
				print('Skipping block at line %d -- <2 lines'%(i_ + 1))
			elif 'Notify' in iblock[0]:
				print('Skipping [Notify*] block starting at line %d'%(i_+1))
			else:
				S_ = parse_block(iblock)
				df_ = pd.concat([df_,S_],axis=1)
			
		# else:
		# 	i_ += 1


	return df_.T


### SPECIFIC USE ###
# Directory Mapping
ROOT = os.path.join('..','..')
OROOT = os.path.join(ROOT,'processed_data','passive','metadata','LOG')
# Get files
DATA = glob(os.path.join(ROOT,'FIELD_2022_2023','SmartSolo_Loc_Processing','LOG_data','*','*','*.LOG'))

for f_ in DATA:
	FPATH,FILE = os.path.split(f_)
	iPATH,ONTS = os.path.split(FPATH)
	jPATH,DAS = os.path.split(iPATH)
	print('=== Processing: %s ==='%(f_))
	# Make write sub-directory, if not already extant
	try:
		os.makedirs(os.path.join(OROOT,DAS,ONTS))
	except:
		pass
	# Read/parse data
	_df_ = getblocks(f_)
	# Write to disk
	_df_.to_csv(os.path.join(OROOT,DAS,ONTS,'Parsed_DigiSolo_LOG.csv'),header=True,index=True)




