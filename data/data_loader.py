import csv


def read_taxi():
    data_list = []
    with open("taxi.csv", newline="") as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            data_list.append(row)
    return data_list


def read_wind():
    data_list = []
    with open("wind.csv", newline="") as data_file:
        reader = csv.reader(data_file)
        for row in reader:
            data_list.append(row)
    return data_list


if __name__ == "__main__":
    taxi_data = read_taxi()
    wind_data = read_wind()
