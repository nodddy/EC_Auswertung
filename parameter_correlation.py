from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


def lin_reggr(x_lst, y_lst):
    x = np.asarray(x_lst).astype(np.float64).reshape(-1, 1)
    y = np.asarray(y_lst).astype(np.float64)
    model = LinearRegression().fit(x, y)
    return model.score(x, y), model.coef_[0], model.intercept_


x_path = Path('C:/Users/MaGoll/Desktop/x_items.txt')
y_path = Path('C:/Users/MaGoll/Desktop/y_items.txt')

x_df = pd.read_csv(x_path, sep='\t')
y_df = pd.read_csv(y_path, sep='\t')

for header_x in list(x_df):
    for header_y in list(y_df):
        lin_fit = lin_reggr(x_df[header_x], y_df[header_y])
        print(header_x, header_y, lin_fit[0])
        if lin_fit[0] > 0.6:
            plt.scatter(x_df[header_x], y_df[header_y])
            plt.plot(x_df[header_x], [i * lin_fit[1] + lin_fit[2] for i in x_df[header_x]])
            plt.show()
