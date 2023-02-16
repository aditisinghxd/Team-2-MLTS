import csv
import pathlib
from matplotlib import pyplot as plt
import numpy as np


def read_taxi():
    """
    Read the taxi csv file and return the contents as a list.
    :return: List of file contents split at new line in file
    """
    path = str(pathlib.Path(__file__).parent)
    data_list = []
    with open(f"{path}\\taxi.csv", newline="") as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            data_list.append(row)
    return data_list


def read_wind():
    """
    Read the wind csv file and return the contents as a list.
    :return: List of file contents split at new line in file
    """
    path = str(pathlib.Path(__file__).parent)
    data_list = []
    with open(f"{path}\\wind.csv", newline="") as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            data_list.append(row)
    return data_list


if __name__ == "__main__":
    """
    Some testing, not relevant for the main program
    """
    taxi_data = read_taxi()
    wind_data = read_wind()

    taxi_array = np.array(taxi_data)
    taxi_values = [int(x) for x in taxi_array[1:, 2]]
    taxi_num = [int(x) for x in taxi_array[1:, 0]]
    plt.scatter(taxi_num, taxi_values, s=1)
    plt.title("Taxi full data")
    plt.show()

    plt.plot(taxi_num[:336], taxi_values[:336])
    plt.legend()
    plt.ylabel("Number of passengers in the 30min bucket")
    plt.xlabel("Sample number in given week")
    plt.show()

    wind_array = np.array(wind_data)
    wind_wind = [float(x) for x in wind_array[1:, 1]]
    wind_num = [x for x in range(0, len(wind_wind))]
    plt.scatter(wind_num, wind_wind, s=1)
    plt.title("Wind speed data")
    plt.show()
