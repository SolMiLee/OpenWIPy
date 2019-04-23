import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# global variable
CURRENT_MEAN = None
VOLTAGE_MEAN = None
CURRENT_STD = None
VOLTAGE_STD = None
POWER_MEAN = None
RESISTANCE_MEAN = None


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


def start_end_time_data(data, start, end, col_time):
    total_time = end - start
    cond = ((data[col_time] >= start) &
            (data[col_time] <= end))
    data = data[cond]

    return data, total_time


# calculate value
def calc_power_value_to_percent(percent=float, power_mean=POWER_MEAN,
                                cur_mean=CURRENT_MEAN, cur_std=CURRENT_STD,
                                vol_mean=VOLTAGE_MEAN, vol_std=VOLTAGE_STD):
    x = np.linspace(-10, 10, 400, dtype=float)
    y = (1 - percent) * power_mean / (vol_std * (x * cur_std + cur_mean)) - vol_mean / vol_std
    return y


def calc_resistance_value_to_percent(percent=float, resistance_mean=RESISTANCE_MEAN,
                                     cur_mean=CURRENT_MEAN, cur_std=CURRENT_STD,
                                     vol_mean=VOLTAGE_MEAN, vol_std=VOLTAGE_STD):
    x = np.linspace(-10, 10, 400, dtype=float)
    y = (1 - percent) * resistance_mean * (x * cur_std + cur_mean) / vol_std - vol_mean / vol_std
    return y


def calc_power_line(percent=float, power_mean=POWER_MEAN,
                    cur_mean=CURRENT_MEAN, cur_std=CURRENT_STD,
                    vol_mean=VOLTAGE_MEAN, vol_std=VOLTAGE_STD, x=float):
    y = (1 - percent) * power_mean / (vol_std * (x * cur_std + cur_mean)) - vol_mean / vol_std
    return y


def clac_resistance_line(percent=float, resistance_mean=RESISTANCE_MEAN,
                         cur_mean=CURRENT_MEAN, cur_std=CURRENT_STD,
                         vol_mean=VOLTAGE_MEAN, vol_std=VOLTAGE_STD, x=float):
    y = (1 - percent) * resistance_mean * (x * cur_std + cur_mean) / vol_std - vol_mean / vol_std
    return y


# plot data, analysis data
def plot_cur_vol(data, col_time, col_current, col_voltage):
    pass_time = data[col_time]
    pass_current = data[col_current]
    pass_voltage = data[col_voltage]

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 6))
    ax[0].plot(pass_time, pass_current, linewidth=0.5)
    ax[0].set_title('Current-Voltage Signal Wave', fontsize=15)
    ax[0].set_ylabel('Current (A)', fontsize=15)
    ax[1].plot(pass_time, pass_voltage, 'k', linewidth=0.5)
    ax[1].set_xlabel('Time (sec)', fontsize=15)
    ax[1].set_ylabel('Voltage (V)', fontsize=15)

    for x in ax:
        x.tick_params(labelsize=10)

    return plt.show()


def plot_cur_vol_with_start_end_time(data, start, end, col_time, col_current, col_voltage, save_data=False):
    d, t = start_end_time_data(data, start, end, col_time)
    if save_data:
        data.to_csv('data.csv')
    plot_cur_vol(d, col_time, col_current, col_voltage)


def plot_cur_vol_distribution_map(data, col_time, col_current, col_voltage):
    cur_mean = round(data[col_current].mean(), 4)
    vol_mean = round(data[col_voltage].mean(), 4)
    cur_std = round(data[col_current].std(), 4)
    vol_std = round(data[col_voltage].std(), 4)
    p_mean = round(cur_mean * vol_mean, 1)
    r_mean = round(vol_mean / cur_mean, 4)

    global CURRENT_MEAN, VOLTAGE_MEAN, CURRENT_STD, VOLTAGE_STD, POWER_MEAN, RESISTANCE_MEAN
    CURRENT_MEAN = cur_mean
    VOLTAGE_MEAN = vol_mean
    CURRENT_STD = cur_std
    VOLTAGE_STD = vol_std
    POWER_MEAN = p_mean
    RESISTANCE_MEAN = r_mean

    watt = []
    resistance = []
    normalized_cur = []
    normalized_vol = []

    for i, v in zip(data[col_current], data[col_voltage]):
        watt.append(round(i * v, 1))

        try:
            resistance.append(round(v / i, 4))
        except ZeroDivisionError:
            resistance.append(0)

        normalized_cur.append((i - cur_mean) / cur_std)
        normalized_vol.append((v - vol_mean) / vol_std)

    plt.figure(figsize=(8, 8))
    plt.scatter(normalized_cur, normalized_vol, s=1, alpha=0.5)
    plt.xlim(left=-10, right=10)
    plt.ylim(bottom=-10, top=10)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Current-Voltage Distribution Map', fontsize=15)
    plt.xlabel('Normalized Current', fontsize=15)
    plt.ylabel('Normalized Voltage', fontsize=15)
    plt.grid(ls='--')

    return plt.show()


def plot_cur_vol_distribution_map_with_per_line():
    pass
