"""
clearn the data and construct the data for training
return csv files for each bus including the following columns:

Features (have been normalized for each bus):
    'Weekday_sin', 'Weekday_cos', 'Hour_sin', 'Hour_cos', 'Temperature (k)', 'Shortwave Radiation (w/m2)',
    'Longwave Radiation (w/m2)', 'Zonal Wind Speed (m/s)', 'Meridional Wind Speed (m/s)', 'Wind Speed (m/s)', 
Target (is not normalized)):    
'Load'
"""

import pandas as pd
from tqdm import trange
import numpy as np
import datetime
import os
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    SAVE_DIR = f'data/data_case{cfg.case.no_bus}/'
    NO_DAY = 365
    NO_BUS_TOTAL = 123
    NO_BUS = cfg.case.no_bus     # number of buses to be selected (should be larger than the targeted number of buses)
    NO_HOUR = 24
    print('Cleaning data for case: ', cfg.case.no_bus)

    # collect load by bus
    load_all = []
    for day in trange(1, NO_DAY+1, desc='Loading load data'):
        load_all.append(pd.read_csv(f'data/Data_public/load_2019/load_annual_D{day}.txt', sep=" ", header=None))
    load_all = pd.concat(load_all, axis=0)
    load_all.reset_index(drop=True, inplace=True)

    # find the buses that are most uncorrelated
    load_corr = np.corrcoef(load_all.values.T)
    bus_index_summary = []
    corr_summary = []
    for i in range(NO_BUS_TOTAL):
        bus_index = [i]
        for _ in range(1, NO_BUS):
            summed_corr = np.sum(load_corr[bus_index, :], axis=0) # sum of correlation of all the previous buses
            j = 0
            new_index = np.argsort(summed_corr)[j]
            while new_index in bus_index:
                j += 1
                new_index = np.argsort(summed_corr)[j]
            bus_index.append(new_index)
        
        bus_index_summary.append(bus_index)
        corr = load_corr[bus_index, :][:, bus_index]
        corr_summary.append(corr.mean())

    index = np.argsort(corr_summary)[0]
    BUS_INDEX = bus_index_summary[index] # the selected bus index

    example_df = pd.read_excel("data/Data_public/Climate_2019/climate_2019_Day" + '1.csv', sheet_name='Hour 1')
    climate_dict = {key: pd.DataFrame(columns=example_df.columns) for key in BUS_INDEX}

    for i in trange(1, NO_DAY+1, desc='Loading climate data'):
        climate_data_all = pd.ExcelFile("data/Data_public/Climate_2019/climate_2019_Day" + str(i) + '.csv')
        for hour in [f'Hour {i}' for i in range(1,NO_HOUR+1)]:
            climate_data_per_hour = climate_data_all.parse(hour)
            for index, bus in enumerate(BUS_INDEX):
                climate_dict[bus] = pd.concat([climate_dict[bus], climate_data_per_hour.iloc[bus-1:bus]], ignore_index=True, axis=0)

    # remove bus index and normalize the climate data
    for bus in BUS_INDEX:
        climate_dict[bus].drop(columns=['Bus'], inplace=True)
        # standardize 
        climate_dict[bus] = (climate_dict[bus] - climate_dict[bus].mean()) / climate_dict[bus].std()

    # add weekday information for each bus
    start_weekday = datetime.datetime(2019,1,1).weekday()
    one_week = np.concatenate([np.arange(start_weekday, 7), (np.arange(0, start_weekday))])

    day = np.repeat(np.arange(1,NO_DAY + 1), 24)
    hour = np.tile(np.arange(1,25), NO_DAY)
    weekday = np.tile(np.repeat(one_week, 24), 53)[:NO_DAY * 24]

    # day_sin = np.sin(2 * np.pi * day / NO_DAY)
    # day_cos = np.cos(2 * np.pi * day / NO_DAY)
    hour_sin = np.sin(2 * np.pi * ( hour / 24))
    hour_cos = np.cos(2 * np.pi * ( hour / 24))
    weekday_sin = np.sin(2 * np.pi * ( weekday / 7))
    weekday_cos = np.cos(2 * np.pi * ( weekday / 7))

    # change the order of the columns
    FEATURE_COLUMNS = ['Weekday_sin', 'Weekday_cos', 'Hour_sin', 'Hour_cos', 'Temperature (k)', 'Shortwave Radiation (w/m2)',
                    'Longwave Radiation (w/m2)', 'Zonal Wind Speed (m/s)',
                    'Meridional Wind Speed (m/s)', 'Wind Speed (m/s)']
    TARGET_COLUMN = ['Load']

    for bus in BUS_INDEX:
        # climate_dict[bus]['Day_sin'] = day_sin
        # climate_dict[bus]['Day_cos'] = day_cos
        climate_dict[bus]['Hour_sin'] = hour_sin
        climate_dict[bus]['Hour_cos'] = hour_cos
        climate_dict[bus]['Weekday_sin'] = weekday_sin
        climate_dict[bus]['Weekday_cos'] = weekday_cos
        climate_dict[bus]['Load'] = load_all[bus]
        climate_dict[bus] = climate_dict[bus][FEATURE_COLUMNS + TARGET_COLUMN]
        climate_dict[bus].reset_index(drop=True, inplace=True)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    for bus in BUS_INDEX:
        climate_dict[bus].to_csv(SAVE_DIR + f'bus_{bus}.csv', index=False)

if __name__ == '__main__':
    main()