# coding: utf-8

import os
import pandas as pd

from glob import glob


def safe_create_dir(dirname):

    try:
        os.makedirs(dirname)
    except OSError:
        pass


def retrieve_filenames(input_path, file_type):

    dir_names = []
    for root, subFolders, files in os.walk(input_path):
        for f in files:
            if file_type in [os.path.splitext(f)[1], ".*"]:
                dir_names += [root]
                break

    dir_names = sorted(dir_names)

    fnames = []
    for dir_name in dir_names:
        dir_fnames = sorted(glob(os.path.join(dir_name, '*' + file_type)))
        fnames += dir_fnames

    return fnames


def fill_missing_data(data):
    if data.notnull().sum() > 1:
        return data.interpolate(method='nearest').ffill().bfill()
    else:
        return data.ffill().bfill()


# Get information of the Airport considering its IATA code
def get_airport_info(iata_code):
    info = pd.read_csv('auxiliar-data/airports.csv', index_col=13, sep=',').loc[iata_code]

    if len(info) == 2:
        info = pd.read_csv('auxiliar-data/airports.csv', index_col=13, sep=',').loc[iata_code].iloc[0]

    return info


# Get the ISO abbreviation for the countries' names
def countries_iso_codes(two_letter_code):
    return pd.read_csv('auxiliar-data/countries_iso_codes.csv', index_col=1, sep=',').loc[two_letter_code]['Alpha-3 code']


# Get the COVID-19 information for the countries take into consideration the IATA code of an airport
def get_covid_info(airport_code, filter_by_date=''):
    two_letter_code = get_airport_info(airport_code)['iso_country']
    three_letter_code = countries_iso_codes(two_letter_code)
    data = pd.read_csv('auxiliar-data/owid-covid-data.csv', index_col=0, sep=',').loc[three_letter_code]

    if filter_by_date:
        return data[data["date"].str.contains(filter_by_date)]

    return data


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



