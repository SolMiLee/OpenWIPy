import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from fractions import Fraction

# area & cumulative num
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
def read_csv_weld_data(data_path, specimen_name, col_names,
                       save_path=None, start=None, end=None):
    '''
    Reading CSV file (Weld raw data must include this columns -> 'Time', 'Current', 'Voltage')
    :param data_path: data path
    :param specimen_name: name of specimen
    :param col_names: list type, basically use ['Time', 'Current', 'Voltage']
    :param save_path: data saving path
    :param start: starting time
    :param end: end time
    :return: if start and end are None return origin data dataframe or return trimed data
    '''

    def start_end_time_data(data, start, end, col_time):
        total_time = end - start
        cond = ((data[col_time] >= start) &
                (data[col_time] <= end))
        data = data[cond]
        print(data)
        print(total_time)
        return data

    raw_data = pd.read_csv(data_path + '\\' + specimen_name + '.csv', header=None, names=col_names, index_col=False)
    raw_data_cleaning = raw_data.dropna(axis=1)
    print(raw_data_cleaning)

    if start is None and end is None:
        return raw_data_cleaning
    else:
        data = start_end_time_data(raw_data_cleaning, start, end, col_names[0])
        data.to_csv(save_path + '\\' + specimen_name + '.csv')
        print('Works Done\n{0} saved\n'.format(specimen_name))
        return data


def read_xl_weld_data(path=str):
    '''
    nothing yet
    :param path: Excel Data Path String
    :return: pass
    '''
    pass


def specify_weld_data_interval(data, step=None):
    step_data = data.iloc[::step]
    print(step_data)
    return step_data


# calculate value
def calc_power_value_to_percent(percent, power_mean,
                                cur_mean, cur_std,
                                vol_mean, vol_std):
    x = np.linspace(-10, 10, 400, dtype=float)
    y = float(1 - percent) * power_mean / float(vol_std * (x * cur_std + cur_mean)) - float(vol_mean / vol_std)
    return y


def calc_resistance_value_to_percent(percent, resistance_mean,
                                     cur_mean, cur_std,
                                     vol_mean, vol_std):
    x = np.linspace(-10, 10, 400, dtype=float)
    y = float(1 - percent) * resistance_mean * (x * cur_std + cur_mean) / vol_std - vol_mean / vol_std
    return y


def calc_power_value(percent, x, power_mean,
                     cur_mean, cur_std,
                     vol_mean, vol_std):
    y = float(1 - percent) * power_mean / float(vol_std * (x * cur_std + cur_mean)) - float(vol_mean / vol_std)
    return y


def calc_resistance_value(percent, x, resistance_mean,
                          cur_mean, cur_std,
                          vol_mean, vol_std):
    y = float(1 - percent) * resistance_mean * (x * cur_std + cur_mean) / vol_std - vol_mean / vol_std
    return y


def calc_scoped_data(data, col_current_name, col_voltage_name,
                     base_df_path, percent):
    cur_mean = round(data[col_current_name].mean(), 4)
    vol_mean = round(data[col_voltage_name].mean(), 4)
    cur_std = round(data[col_current_name].std(), 4)
    vol_std = round(data[col_voltage_name].std(), 4)
    p_mean = round(cur_mean * vol_mean, 1)
    r_mean = round(vol_mean / cur_mean, 4)

    pos_power = []
    neg_power = []
    pos_resistance = []
    neg_resistance = []

    df = pd.read_csv(base_df_path + '\\' + 'base_df.csv')

    for nc in df['Normalized Current']:
        pos_power.append(calc_power_value(-percent, nc, p_mean, cur_mean, cur_std, vol_mean, vol_std))
        neg_power.append(calc_power_value(percent, nc, p_mean, cur_mean, cur_std, vol_mean, vol_std))
        pos_resistance.append(calc_resistance_value(-percent, nc, r_mean, cur_mean, cur_std, vol_mean, vol_std))
        neg_resistance.append(calc_resistance_value(percent, nc, r_mean, cur_mean, cur_std, vol_mean, vol_std))

    power_condition = ((df['Normalized Voltage'] >= neg_power) &
                       (df['Normalized Voltage'] <= pos_power))
    resistance_condition = ((df['Normalized Voltage'] >= neg_resistance) &
                            (df['Normalized Voltage'] <= pos_resistance))

    power_data = df[power_condition]
    resistance_data = df[resistance_condition]
    power_and_resistance_data = pd.merge(power_data, resistance_data)
    return power_and_resistance_data


# plot data, analysis data
def plot_cur_vol(data, col_time_name, col_current_name, col_voltage_name):
    '''
    plotting current & voltage wave from weld data
    :param data: data file
    :param col_time_name: column name str
    :param col_current_name: column name str
    :param col_voltage_name: column name str
    :return: matplotlib.pyplot.show() plotting current voltage wave
    '''
    pass_time = data[col_time_name]
    pass_current = data[col_current_name]
    pass_voltage = data[col_voltage_name]

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


def plot_cur_vol_distribution_map(data, col_time_name, col_current_name, col_voltage_name,
                                  percent=None, withline=False):
    '''
    :param data: weld data
    :param col_time_name: column name
    :param col_current_name: column name
    :param col_voltage_name: column name
    :param percent: percent :)
    :param withline: True -> including Power and Resistance line
    :return: matplotlib.pyplot.show() Plot Current Voltage Distribution Map
    '''
    cur_mean = round(data[col_current_name].mean(), 4)
    vol_mean = round(data[col_voltage_name].mean(), 4)
    cur_std = round(data[col_current_name].std(), 4)
    vol_std = round(data[col_voltage_name].std(), 4)
    p_mean = round(cur_mean * vol_mean, 1)
    r_mean = round(vol_mean / cur_mean, 4)

    watt = []
    resistance = []
    normalized_cur = []
    normalized_vol = []

    for i, v in zip(data[col_current_name], data[col_voltage_name]):
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

    if withline:
        xr = np.linspace(-10, 10, 400, dtype=float)

        power_y1 = calc_power_value_to_percent(-percent, p_mean, cur_mean, cur_std, vol_mean, vol_std)
        power_y2 = calc_power_value_to_percent(percent, p_mean, cur_mean, cur_std, vol_mean, vol_std)
        resistance_y1 = calc_resistance_value_to_percent(-percent, r_mean, cur_mean, cur_std, vol_mean, vol_std)
        resistance_y2 = calc_resistance_value_to_percent(percent, r_mean, cur_mean, cur_std, vol_mean, vol_std)

        power_line1, = plt.plot(xr, power_y1, 'k', linewidth=2)
        power_line2, = plt.plot(xr, power_y2, 'k--', linewidth=2)
        resistance_line1, = plt.plot(xr, resistance_y1, 'r:', linewidth=2)
        resistance_line2, = plt.plot(xr, resistance_y2, 'r-.', linewidth=2)

        plt.legend((power_line1, resistance_line1, power_line2, resistance_line2),
                   ('P1 %.1f' % (p_mean * (1 + percent)),
                    'R1 %.4f' % (r_mean * (1 + percent)),
                    'P2 %.1f' % (p_mean * (1 - percent)),
                    'R2 %.4f' % (r_mean * (1 - percent))),
                   fontsize=15, loc='lower right')

    return plt.show()


# make dataframe
def make_base_dataframe(data, col_cur_name, col_vol_name, save_path):
    cur_mean = round(data[col_cur_name].mean(), 4)
    vol_mean = round(data[col_vol_name].mean(), 4)
    cur_std = round(data[col_cur_name].std(), 4)
    vol_std = round(data[col_vol_name].std(), 4)
    p_mean = round(cur_mean * vol_mean, 1)
    r_mean = round(vol_mean / cur_mean, 4)

    watt = []
    ohm = []
    normalized_cur = []
    normalized_vol = []

    for i, v in zip(data[col_cur_name], data[col_vol_name]):
        watt.append(round(i * v, 1))

        try:
            ohm.append(round(v / i, 4))
        except ZeroDivisionError:
            ohm.append(0)

        normalized_cur.append((i - cur_mean) / cur_std)
        normalized_vol.append((v - vol_mean) / vol_std)

    d = {'Time': data['Time'],
         'Current': round(data['Current'], 4), 'Voltage': round(data['Voltage'], 4),
         'Watt': watt, 'Resistance': ohm,
         'Normalized Current': normalized_cur, 'Normalized Voltage': normalized_vol}
    df = DataFrame(data=d, dtype=float)
    df.to_csv(save_path + '\\' + 'base_df.csv', index=False)
    print('base_df saved')
    return df


def make_distribution_count_dataframe(after_data, path, area_path, cumulative_path):
    data_len = len(after_data)

    a_count = []
    c_count = []
    a_per = []
    c_per = []

    for n in np.arange(50):
        ad = pd.read_csv(area_path + '\\' + area[n] + '.csv')
        cd = pd.read_csv(cumulative_path + '\\' + cumulative[n] + '.csv')
        a_count.append(len(ad))
        c_count.append(len(cd))
        a_per.append(round(len(ad) / data_len, 3))
        c_per.append(round(len(cd) / data_len, 3))

    d = {'Area Count': a_count, 'Cumulative Count': c_count,
         'Area Per': a_per, 'Cumulative Per': c_per}

    df = DataFrame(data=d)
    df.to_csv(path + '\\' + 'distribution_df.csv')
    print(df)
    return df


def make_area_cumulated_df_to_csv(after_data, base_df_path,
                                  area_folder_path, cumulative_folder_path,
                                  whichis=('area', 'cumulative')):
    '''
    :param after_data:
    :param area_folder_path:
    :param cumulative_folder_path:
    :param whichis: choice 'area' or 'cumulative'
    :return: None
    '''

    pct = np.arange(0.02, 1.02, 0.02)

    for n in np.arange(0, 50):
        if whichis == 'area' and n == 0:
            df0 = calc_scoped_data(after_data, 'Current', 'Voltage', base_df_path, pct[0])
            df0.to_csv(area_folder_path + '\\' + area[0] + '.csv', index=False)
            print('a00 saved')

        elif whichis == 'area' and n >= 1:
            df = pd.concat([calc_scoped_data(after_data, 'Current', 'Voltage', base_df_path, pct[n]),
                            calc_scoped_data(after_data, 'Current', 'Voltage', base_df_path, pct[n - 1])],
                           sort=False).drop_duplicates(keep=False)
            df.to_csv(area_folder_path + '\\' + area[n] + '.csv', index=False)
            print('{0} saved'.format(area[n]))

        elif whichis == 'cumulative':
            df = calc_scoped_data(after_data, 'Current', 'Voltage', base_df_path, pct[n])
            df.to_csv(cumulative_folder_path + '\\' + cumulative[n] + '.csv', index=False)
            print('{0} saved'.format(cumulative[n]))
        else:
            print('Wrong Parameter')
    print('Works Done')
