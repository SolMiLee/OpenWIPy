import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# data modify
def read_csv_weld_data(path=str, col_nm=None):
    raw_data = pd.read_csv(path, header=None, names=col_nm, index_col=False)
    raw_data_cleaning = raw_data.dropna(axis=1)
    return raw_data_cleaning


def read_xl_weld_data(path=str):
    pass


def specify_weld_data_interval(data, step=None):
    step_data = data.iloc[::step]
    return step_data


# plot data
def plot_current_voltage(col_time, col_current, col_voltage):
    pass_time = col_time
    pass_current = col_current
    pass_voltage = col_voltage

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 6))
    ax[0].plot(pass_time, pass_current, linewidth=0.5)
    ax[0].set_title('Current, Voltage Signal')
    ax[0].set_ylabel('Current (A)', fontsize=15)
    ax[1].plot(pass_time, pass_voltage, 'k', linewidth=0.5)
    ax[1].set_xlabel('Time (sec)', fontsize=15)
    ax[1].set_ylabel('Voltage (V)', fontsize=15)

    for x in ax:
        x.tick_params(labelsize=10)

    return plt.show()
