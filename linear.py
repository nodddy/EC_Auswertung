from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

file = Path(
    'N:/BZ_Neu/23_char_hp/5_Ergebnisse/MGo/Masterarbeit/Daten/RDE/20190911_3FeCo-N-C_E3/analysis/20190911_3FeCo-E3_ORR_before_cathodic.txt')


def slice_df(df, window=1):
    current_index = df.first_valid_index()
    slice_list = []
    while current_index < len(df):
        slice_list.append(df.iloc[current_index:current_index + window])
        current_index += window
    return slice_list


def linear_regr(df):
    """ applies linear reggression to x and y values of tuple list and return slope value """

    x_lst = df['Potential [V]'].to_numpy()
    y_lst = df['Disk Current [A]'].to_numpy()
    X = np.asarray(x_lst).astype(np.float64).reshape(-1, 1)
    Y = np.asarray(y_lst).astype(np.float64)
    model = LinearRegression().fit(X, Y)
    return model.coef_[0], model.intercept_, model.score(X, Y)


def get_linear(input_list):
    def modifiy_list(input_list):
        def compare(lst):
            def comp_fnc(item1, item2):
                """ item is tuple ([slope, intercept, r_square], dataframe) """

                df = pd.DataFrame()
                df['Potential [V]'] = pd.concat([item1[1]['Potential [V]'], item2[1]['Potential [V]']])
                df['Reggr'] = df['Potential [V]'] * item1[0][0] + item1[0][1]
                df['Delta1'] = abs(abs(item1[1]['Disk Current [A]']) - abs(df['Reggr']))
                delta_max = df['Delta1'].max() * 2.5
                df['Delta2'] = abs(abs(item2[1]['Disk Current [A]']) - abs(df['Reggr']))
                outlier_list = df.query('Delta2 > @delta_max', inplace=False)
                df.reset_index(inplace=True)
                if len(outlier_list) <= len(df)/20:
                    return True
                else:
                    return False

            def comp_fnc2(item1, item2):
                pass

            try:
                if comp_fnc(lst[0], lst[1]) is True:
                    new_df = pd.concat([lst[0][1], lst[1][1]], verify_integrity=True, ignore_index=True)
                    new_slope, new_intercept, new_r_sqr = linear_regr(new_df)
                    return (new_slope, new_intercept, new_r_sqr), new_df
                else:
                    return None
            except IndexError:
                return None

        comp = compare(input_list)
        output_list = input_list.copy()
        if comp is not None:
            output_list.pop(0)
            output_list.pop(0)
            output_list.insert(0, (comp[0], comp[1]))
            return True, output_list, None
        else:
            linear = output_list.pop(0)
            return False, output_list, linear

    linear_list = []
    lst = input_list.copy()

    while True:
        if len(lst) < 1:
            return linear_list
        mod = modifiy_list(lst)
        lst = mod[1]
        if mod[0] is False:
            linear_list.append(mod[2])


data = pd.read_csv(file, sep='\t')
data['Disk Current [A]'] = data['Disk Current [A]'].ewm(span=50).mean()
data = data.rolling(window=20, win_type='gaussian', center=True).mean(std=0.5).dropna()

slice_lst = slice_df(data, 50)
slope_lst = [linear_regr(item) for item in slice_lst]
zipped_list = list(zip(slope_lst, slice_lst))
#lin_lst = get_linear(zipped_list)
lin_lst=[]
for item in lin_lst:
    if item[0][2]>0.9:
        plt.plot(item[1]['Potential [V]'], item[1]['Disk Current [A]'])




max_j=data['Disk Current [A]'].min()
data['Disk']=abs(data['Disk Current [A]'])
onset_threshhold=abs(max_j*0.02)
onset=data.query('Disk > @onset_threshhold', inplace=False)
print(onset.tail(1))
plt.scatter(onset.tail(1)['Potential [V]'], onset.tail(1)['Disk Current [A]'])
plt.plot(data['Potential [V]'], data['Disk Current [A]'])


plt.show()
