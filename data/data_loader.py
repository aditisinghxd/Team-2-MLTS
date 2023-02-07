import csv
import pathlib


def read_taxi():
    path = str(pathlib.Path(__file__).parent)
    data_list = []
    with open(f"{path}\\taxi.csv", newline="") as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            data_list.append(row)
    return data_list


def read_wind():
    path = str(pathlib.Path(__file__).parent)
    data_list = []
    with open(f"{path}\\wind.csv", newline="") as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            data_list.append(row)
    return data_list


if __name__ == "__main__":
    taxi_data = read_taxi()
    wind_data = read_wind()
