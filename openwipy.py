import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_csv_weld_data(path=str, col_nm=None):
    raw_data = pd.read_csv(path, header=None, names=col_nm, index_col=False)
    raw_data_cleaning = raw_data.dropna(axis=1)
    return raw_data_cleaning


def plot_current_voltage(raw_data):
    pass_time = raw_data['Time']
    pass_current = raw_data['Current']
    pass_voltage = raw_data['Voltage']

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 6))
    ax[0].plot(pass_time, pass_current, linewidth=0.5)
    ax[0].set_ylabel('Current (A)', fontsize=20)
    ax[1].plot(pass_time, pass_voltage, 'k', linewidth=0.5)
    ax[1].set_xlabel('Time (sec', fontsize=20)
    ax[1].set_ylabel('Voltage (V)', fontsize=20)

    for x in ax:
        x.tick_params(labelsize=15)

    return plt.show()

# ReadCSVData('C:\Prjt\OpenWIPy\data\S01P04A.csv')
# rdata = ReadCSVData('C:\Prjt\OpenWIPy\data\S01P04A.csv')
# PlotCV(rdata)
