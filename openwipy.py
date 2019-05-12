import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

# global variable
CURRENT_MEAN = None
VOLTAGE_MEAN = None
CURRENT_STD = None
VOLTAGE_STD = None
POWER_MEAN = None
RESISTANCE_MEAN = None
# SPECIMEN_NAME = None

area = ['a00', 'a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09',
        'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19',
        'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29',
        'a30', 'a31', 'a32', 'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39',
        'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48', 'a49']

cumulative = ['c00', 'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09',
              'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19',
              'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29',
              'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39',
              'c40', 'c41', 'c42', 'c43', 'c44', 'c45', 'c46', 'c47', 'c48', 'c49']


# data modify
def read_csv_weld_data(path=str, col_nm=None):
    raw_data = pd.read_csv(path, header=None, names=col_nm, index_col=False)
    raw_data_cleaning = raw_data.dropna(axis=1)
    return raw_data_cleaning


def read_xl_weld_data(path=str):
    '''
    :param path:Excel Data Path String
    :return:None
    '''
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


def calc_resistance_line(percent=float, resistance_mean=RESISTANCE_MEAN,
                         cur_mean=CURRENT_MEAN, cur_std=CURRENT_STD,
                         vol_mean=VOLTAGE_MEAN, vol_std=VOLTAGE_STD, x=float):
    y = (1 - percent) * resistance_mean * (x * cur_std + cur_mean) / vol_std - vol_mean / vol_std
    return y


def calc_scoped_data(specimen_nm, percent=float):
    pos_power = []
    neg_power = []
    pos_resistance = []
    neg_resistance = []

    path = '..\\' + specimen_nm + '\\'
    df = pd.read_csv(path + 'base_df.csv')

    for nc in df['Normalized Current']:
        pos_power.append(calc_power_line(-percent, nc))
        neg_power.append(calc_power_line(percent, nc))
        pos_resistance.append(calc_resistance_line(-percent, nc))
        neg_resistance.append(calc_resistance_line(percent, nc))

    power_condition = ((df['Normalized Voltage'] >= neg_power) &
                       (df['Normalized Voltage'] <= pos_power))
    resistance_condition = ((df['Normalized Voltage'] >= neg_resistance) &
                            (df['Normalized Votlage'] <= pos_resistance))

    power_data = df[power_condition]
    resistance_data = df[resistance_condition]
    power_and_resistance_data = pd.merge(power_data, resistance_data)
    return power_and_resistance_data


# plot data, analysis data
def plot_cur_vol(data, col_time, col_current, col_voltage):
    '''
    :param data:
    :param col_time:
    :param col_current:
    :param col_voltage:
    :return:
    '''
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
    '''
    :param data:
    :param col_time:
    :param col_current:
    :param col_voltage:
    :return: matplotlib.pyplot.show() Plot Current Voltage Distribution Map
    '''
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


def plot_cur_vol_distribution_map_with_per_line(data, percent, col_time, col_current, col_voltage):
    plot_cur_vol_distribution_map(data, col_time, col_current, col_voltage)

    xr = np.linspace(-10, 10, 400, dtype=float)

    power_y1 = calc_power_value_to_percent(-percent)
    power_y2 = calc_power_value_to_percent(percent)
    resistance_y1 = calc_resistance_value_to_percent(-percent)
    resistance_y2 = calc_resistance_value_to_percent(percent)

    power_line1, = plt.plot(xr, power_y1, 'k', linewidth=5)
    power_line2, = plt.plot(xr, power_y2, 'k--', linewidth=5)
    resistance_line1, = plt.plot(xr, resistance_y1, 'r', linewidth=5)
    resistance_line2, = plt.plot(xr, resistance_y2, 'r--', linewidth=5)

    plt.legend((power_line1, resistance_line1, power_line2, resistance_line2),
               ('P1 %.1f' % (POWER_MEAN * (1 + percent)),
                'R1 %.4f' % (RESISTANCE_MEAN * (1 + percent)),
                'P2 %.1f' % (POWER_MEAN * (1 - percent)),
                'R2 %.4f' % (RESISTANCE_MEAN * (1 - percent))),
               fonsize=15, loc='lower right')


# make dataframe
def make_base_dataframe(specimen_nm, data, watt, ohm, normalized_cur, normalized_vol):
    d = {'Time': data['Time'],
         'Current': round(data['Current'], 4), 'Voltage': round(data['Voltage'], 4),
         'Watt': watt, 'Resistance': ohm,
         'Normalized Current': normalized_cur, 'Normalized Voltage': normalized_vol}
    path = '..\\' + specimen_nm + '\\'
    df = DataFrame(data=d, dtype=float)
    df.to_csv(path + 'base_df.csv', index=False)
    return df


def make_distribution_count_dataframe(specimen_nm, data):
    area_path = '..\\' + specimen_nm + '\\area\\'
    cumulative_path = '..\\' + specimen_nm + '\\cumulative\\'

    data_len = len(data)

    a_count = []
    c_count = []
    a_per = []
    c_per = []

    for n in np.arange(50):
        ad = pd.read_csv(area_path + area[n] + '.csv')
        cd = pd.read_csv(cumulative_path + cumulative[n] + '.csv')
        a_count.append(len(ad))
        c_count.append(len(cd))
        a_per.append(round(len(ad) / data_len, 3))
        c_per.append(round(len(cd) / data_len, 3))

    d = {'Area Count': a_count, 'Cumulative Count': c_count,
         'Area Per': a_per, 'Cumulative Per': c_per}

    path = '..\\' + specimen_nm + '\\'

    df = DataFrame(data=d)
    df.to_csv(path + 'distribution_df.csv')
    return df


def make_area_cumulated_df_to_csv(speciemn_nm, whichis=('area', 'cumulative')):
    '''
    :param speciemn_nm: Specimen name string
    :param whichis: 'area' or 'cumulative'
    :return: None
    '''
    area_path = '..\\' + speciemn_nm + '\\area\\'
    cumulative_path = '..\\' + speciemn_nm + '\\cumulative\\'

    pct = np.arange(0.02, 1.02, 0.02)

    for n in np.arange(0, 50):
        if whichis == 'area':
            df0 = calc_scoped_data(speciemn_nm, pct[0])
            df0.to_csv(area_path + area[0] + '.csv', index=False)

            for n in np.arange(1, 50):
                df = pd.concat([calc_scoped_data(speciemn_nm, pct[n]), calc_scoped_data(speciemn_nm, pct[n - 1])],
                               sort=False).drop_duplicates(keep=False)
                df.to_csv(area_path + area[n] + '.csv', index=False)

        elif whichis == 'cumulative':
            df = calc_scoped_data(speciemn_nm, pct[n])
            df.to_csv(cumulative_path + cumulative[n] + '.csv', index=False)
        else:
            print('Wrong Parameter')
