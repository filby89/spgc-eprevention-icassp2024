import os
import datetime
import pandas as pd
import numpy as np
import pyhrv
import scipy
import argparse

valid_ranges = {
	"acc_X" : (-19.6, 19.6),
	"acc_Y" : (-19.6, 19.6),
	"acc_Z" : (-19.6, 19.6),
	"gyr_X" : (-573, 573),
	"gyr_Y" : (-573, 573),
	"gyr_Z" : (-573, 573),
	"heartRate" : (0, 255),
	"rRInterval" : (0, 2000),
}

def rmssd(x):
	x = x.dropna()
	try:
		rmssd = pyhrv.time_domain.rmssd(x)[0]
	except (ZeroDivisionError, ValueError):
		rmssd = np.nan

	return rmssd

def sdnn(x):
	x = x.dropna()
	try:
		sdnn = pyhrv.time_domain.sdnn(x)[0]
	except (ZeroDivisionError, ValueError):
		sdnn = np.nan

	return sdnn


def lombscargle_power_high(nni):
	# high frequencies
	l = 0.15 * np.pi /2
	h = 0.4 * np.pi /2
	freqs = np.linspace(l, h, 1000)
	hf_lsp = scipy.signal.lombscargle(nni.to_numpy(), nni.index.to_numpy(), freqs, normalize=True)
	return np.trapz(hf_lsp, freqs)


def get_norm(df):
	""" Returns the mean norm of the x,y,z columns of a dataframe"""
	df = df.dropna()
	return np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).mean() 


def time_encoding(slice):
	# Compute the sin and cos of timestamp (we have 12*24=288 5-minutes per day)
	mean_timestamp = slice['timecol'].astype('datetime64').mean()
	h = mean_timestamp.hour
	m = mean_timestamp.minute
	time_value = h*60 + m
	sin_t = np.sin(time_value*(2.*np.pi/(60*24)))
	cos_t = np.cos(time_value*(2.*np.pi/(60*24)))
	return sin_t, cos_t


# function that does feature extraction for a patient
def extract_user_features(patient, phase, dataset_path, features_path):
	print('Extracting features for patient {} and phase {}'.format(patient, phase))
	if os.path.exists(f'{dataset_path}/{patient}/{phase}/linacc.parquet'):
		df_linacc = pd.read_parquet(f'{dataset_path}/{patient}/{phase}/linacc.parquet')

		df_linacc['DateTime'] = df_linacc['time'].apply(lambda t: datetime.datetime.combine(datetime.datetime.today(), t))

		# where acc is out of limits, set it to nan
		df_linacc.loc[(df_linacc['X'] < valid_ranges['acc_X'][0]) | (df_linacc['X'] >= valid_ranges['acc_X'][1]), 'X'] = np.nan
		df_linacc.loc[(df_linacc['Y'] < valid_ranges['acc_Y'][0]) | (df_linacc['Y'] >= valid_ranges['acc_Y'][1]), 'Y'] = np.nan
		df_linacc.loc[(df_linacc['Z'] < valid_ranges['acc_Z'][0]) | (df_linacc['Z'] >= valid_ranges['acc_Z'][1]), 'Z'] = np.nan

		df_linacc = df_linacc.groupby([df_linacc['day_index'],pd.Grouper(key='DateTime',freq='5Min')]).apply(get_norm)

	# if os.path.exists(f'{dataset_path}/{patient}/{phase}/gyr.parquet'): use this if you want to extract gyr features
	# 	df_gyr = pd.read_parquet(f'{dataset_path}/{patient}/{phase}/gyr.parquet')

	# 	df_gyr['DateTime'] = df_gyr['time'].apply(lambda t: datetime.datetime.combine(datetime.datetime.today(), t))

	# 	# where gyr is out of limits, set it to nan
	# 	df_gyr.loc[(df_gyr['X'] < valid_ranges['gyr_X'][0]) | (df_gyr['X'] >= valid_ranges['gyr_X'][1]), 'X'] = np.nan
	# 	df_gyr.loc[(df_gyr['Y'] < valid_ranges['gyr_Y'][0]) | (df_gyr['Y'] >= valid_ranges['gyr_Y'][1]), 'Y'] = np.nan
	# 	df_gyr.loc[(df_gyr['Z'] < valid_ranges['gyr_Z'][0]) | (df_gyr['Z'] >= valid_ranges['gyr_Z'][1]), 'Z'] = np.nan
		
	# 	df_gyr = df_gyr.groupby([df_gyr['day_index'],pd.Grouper(key='DateTime',freq='5Min')]).apply(get_norm)

	if os.path.exists(f'{dataset_path}/{patient}/{phase}/hrm.parquet'):
		df_hrm = pd.read_parquet(f'{dataset_path}/{patient}/{phase}/hrm.parquet')

		df_hrm['DateTime'] = df_hrm['time'].apply(lambda t: datetime.datetime.combine(datetime.datetime.today(), t))

		# where hearRate is out of limits, set it to nan
		df_hrm.loc[df_hrm['heartRate'] <= valid_ranges['heartRate'][0], 'heartRate'] = np.nan
		df_hrm.loc[df_hrm['heartRate'] > valid_ranges['heartRate'][1], 'heartRate'] = np.nan

		# same for rRInterval
		df_hrm.loc[df_hrm['rRInterval'] <= valid_ranges['rRInterval'][0], 'rRInterval'] = np.nan
		df_hrm.loc[df_hrm['rRInterval'] > valid_ranges['rRInterval'][1], 'rRInterval'] = np.nan
	
		df_hrm = df_hrm.groupby([df_hrm['day_index'],pd.Grouper(key='DateTime',freq='5Min')]).agg({'heartRate': np.nanmean, 'rRInterval':  [np.nanmean, rmssd, sdnn, lombscargle_power_high]})


	# combine all
	df = pd.concat([df_linacc, df_hrm], axis=1)
	df = df.reset_index()
	
	h = df['DateTime'].dt.hour
	m = df['DateTime'].dt.minute
	time_value = h*60 + m
	df['sin_t'] = np.sin(time_value*(2.*np.pi/(60*24)))
	df['cos_t'] = np.cos(time_value*(2.*np.pi/(60*24)))
	
	# drop datetime column
	df = df.drop(columns=['DateTime'])

	# rename columns
	new_column_names = ['day_index', 'acc_norm', 'heartRate_mean', 'rRInterval_mean', 'rRInterval_rmssd', 'rRInterval_sdnn', 'rRInterval_lombscargle_power_high', 'sin_t', 'cos_t']
	df.columns = new_column_names

	# save df
	os.makedirs(f'{features_path}/{patient}/{phase}', exist_ok=True)
	df.to_csv(f'{features_path}/{patient}/{phase}/features.csv')
	print('Saved features for patient {} and phase {}'.format(patient, phase))



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_path', type=str, required=True, help='path to raw downloaded data')
	parser.add_argument('--out_features_path', type=str, required=True)
	parser.add_argument('--n_jobs', type=int, default=8)
	
	args = parser.parse_args()



	patients = os.listdir(args.dataset_path)
	combs = []
	for patient in patients:
		for phase in os.listdir(os.path.join(args.dataset_path,patient)):
			combs.append([patient, phase])

	from joblib import Parallel, delayed
	Parallel(n_jobs=args.n_jobs)(delayed(extract_user_features)(patient, phase, args.dataset_path, args.out_features_path) for patient, phase in combs)

