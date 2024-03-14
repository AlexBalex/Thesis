from scipy.io import loadmat
import pandas as pd

mat_contents = loadmat('1_20160518.mat')

session_name = 'cz_eeg1'  # name of the session you want to convert (cz__eeg1 - cz_eeg24)
session_data = mat_contents.get(session_name)

if session_data is None:
    raise ValueError(f"No data found for session '{session_name}'")

dataframes_list = []

for i, array in enumerate(session_data):

    array_df = pd.DataFrame(array)
    

    array_df.columns = [f"{i}_{col}" for col in array_df.columns]
    
    dataframes_list.append(array_df)

all_arrays_df = pd.concat(dataframes_list, axis=1)

csv_filename = f'{session_name}_data.csv'
all_arrays_df.to_csv(csv_filename, index=False)