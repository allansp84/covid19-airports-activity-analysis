# coding: utf-8

import os
import sys
import operator
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import breakout_detection
import multiprocessing

from tqdm import tqdm
from glob import glob
from datetime import date
from datetime import datetime

from scipy import stats
from sklearn import linear_model
from sklearn import metrics
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

from common import *



def simple_moving_average(df, wsize=3):
    # Computes the moving average considering a panda Dataframe as input. The parameter wsize corresponds to the size of the window used during computation.

    smas = []
    dates = []
    for i in range(wsize, df.shape[0]):
        dates += [df.index[i]]
        smas += [df.iloc[i - wsize:i].sum() / wsize]

    sma_df = pd.concat(smas, axis=1).transpose()
    sma_df = sma_df.set_index(pd.Index(dates))

    return sma_df


def simple_moving_average_covid(df, wsize=3):
    # Compute the moving average considering the COVID-19 CSV file provided along with this code named as owid-covid-data.csv, which is available for download in the https://ourworldindata.org/coronavirus-source-data

    columns_name = ['total_cases', 'total_deaths', 'new_cases', 'new_deaths']
    columns_idxs = sorted([df.columns.get_loc(c) for c in columns_name if c in df])

    smas = []
    dates = []
    for i in range(wsize, df.shape[0]):
        dates += [df.iloc[:, 2][i - 1:i]]
        smas += [df.iloc[:, columns_idxs][i - wsize:i].sum() / wsize]

    sma_df = pd.concat(smas, axis=1).transpose()
    sma_df = sma_df.set_index(pd.DatetimeIndex(pd.concat(dates)))

    return sma_df


def breakout_detection_sma_crossover(df, sma, short_term_ws, long_term_ws):

    sma_long_term = simple_moving_average(df, wsize=short_term_ws + long_term_ws)
    diff = sma[long_term_ws:] < sma_long_term
    diff_forward = diff.shift(1)
    crossing = np.where(abs(diff - diff_forward) == 1)[0]
    breaks = sma.iloc[crossing][sma.iloc[crossing].index > '2019-07'].idxmin(axis=0)

    return breaks



def breakout_detection_twitter(df, sma, airport_names, ws, msize, beta):
    # N. A. James, A. Kejariwal and D. S. Matteson, "Leveraging cloud data to mitigate user experience from ‘breaking bad’," 2016 IEEE International Conference on Big Data (Big Data), Washington, DC, 2016, pp. 3499-3508, doi: 10.1109/BigData.2016.7841013.
    # output_path = "plots"
    # safe_create_dir(os.path.join(output_path, 'twitter-algo', str(ws)))

    breaks = []
    for name in airport_names:
        edm_multi = breakout_detection.EdmMulti()
        edm_multi.evaluate(sma[name], min_size=int(msize), beta=float(beta), degree=1)
        break_idxs = list(edm_multi.getLoc())

        if len(break_idxs) == 0:
            break_idxs = [len(sma[name]) - 1]

        breaks += [sma[name].index[break_idxs[-1]]]

        # plt.clf()
        # df[name].plot(figsize=(30,10))
        # for i in edm_multi.getLoc(): 
        #     plt.axvline(df[name].index[i], color='#FF4E24')

        # out_path = os.path.join(output_path, "twitter-algo", str(ws), "breaks-msize-{}-beta-{}".format(msize, beta))
        # safe_create_dir(out_path)
        # plt.savefig(os.path.join(out_path, "{}.png".format(name)))
        # plt.close('all')

    breaks = pd.Series(breaks, index=airport_names)

    return breaks


def run_trial_breakout_sma_crossover(data, st_ws, lt_ws, output_path):

    outpath = os.path.join(output_path, 'trials', 'st_ws-{}'.format(st_ws), 'lt_ws-{}'.format(lt_ws))

    safe_create_dir(outpath)

    sma = simple_moving_average(data, wsize=st_ws)
    breaking_points = breakout_detection_sma_crossover(data, sma, st_ws, lt_ws)

    # -- compute residual values
    residuals = (breaking_points - datetime(2020, 5, 1)).astype('timedelta64[D]')
    residuals = residuals.append(pd.Series({'MAE': np.mean(np.abs(residuals))}))
    residuals = residuals.append(pd.Series({'RMSE': np.sqrt(np.mean(np.power(residuals, 2)))}))

    # -- save residual values
    residuals[::-1].to_csv(os.path.join(outpath, 'residuals.txt'), sep='\t', header=False)

    # -- save the detected breaking points
    breaking_points.to_csv(os.path.join(outpath, 'breaking_points.txt'), sep='\t', header=False)

    return [st_ws, lt_ws, np.mean(np.abs(residuals)), np.sqrt(np.mean(np.power(residuals, 2)))]


def run_trial_breakout_twitter(data, airport_names, ws, msize, beta, output_path):

    outpath = os.path.join(output_path, 'trials', 'ws-{}'.format(ws), 'msize-{}'.format(msize), 'beta-{}'.format(beta))

    safe_create_dir(outpath)

    sma = simple_moving_average(data, wsize=ws)
    breaking_points = breakout_detection_twitter(data, sma, airport_names, ws, msize, beta)

    # -- compute residual values
    residuals = (breaking_points - datetime(2020, 5, 1)).astype('timedelta64[D]')
    residuals = residuals.append(pd.Series({'MAE': np.mean(np.abs(residuals))}))
    residuals = residuals.append(pd.Series({'RMSE': np.sqrt(np.mean(np.power(residuals, 2)))}))

    # -- save residual values
    residuals[::-1].to_csv(os.path.join(outpath, 'residuals.txt'), sep='\t', header=False)

    # -- save the detected breaking points
    breaking_points.to_csv(os.path.join(outpath, 'breaking_points.txt'), sep='\t', header=False)

    return [ws, msize, beta, np.mean(np.abs(residuals)), np.sqrt(np.mean(np.power(residuals, 2)))]


def find_parameters_breakout_detection_twitter(data, airport_names, output_path, n_threads):

    trials_fname = os.path.join(output_path, "trials_results.txt")
    if not os.path.exists(trials_fname):

        # -- parameter space
        betas = np.linspace(1e-1, 0.9, 9)
        window_sizes = np.arange(7, 49 + 1, 7)
        msizes = np.arange(64, 128 + 1, 8)

        # -- prepare trials
        trails = [[data, airport_names, ws, msize, beta, output_path] for ws in window_sizes for msize in msizes for beta in betas]

        # -- run trials in parallel
        trials_results = Parallel(n_jobs=n_threads)(
            delayed(run_trial_breakout_twitter)(*trial) for trial in tqdm(trails, desc="Breakout Detection (Twitter's algorithm)"))

        # -- save results for all trials in disk
        trials_results = sorted(trials_results, key=operator.itemgetter(4), reverse=False)
        np.savetxt(trials_fname, trials_results, fmt="%d,%d,%f,%f,%f")

    else:
        # -- ready the trials results from a previous execution
        trials_results = pd.read_csv(trials_fname, sep=',', header=None)
        trials_results = list(trials_results.to_records(index=False))

    trials_results = sorted(trials_results, key=operator.itemgetter(3), reverse=False)
    print(bcolors.OKGREEN, "Best configuration:", trials_results[0], bcolors.ENDC)

    return trials_results


def find_parameters_breakout_detection_sma_crossover(data, airport_names, output_path, n_threads):

    trials_fname = os.path.join(output_path, "trials_results.txt")
    if not os.path.exists(trials_fname):

        # -- parameter space
        st_window_sizes = np.arange(7, 49 + 1, 7)
        lt_window_sizes = np.arange(7, 49 + 1, 7)

        # -- prepare trials
        trails = [[data, st_ws, lt_ws, output_path] for st_ws in st_window_sizes for lt_ws in lt_window_sizes]

        # -- run trials in parallel
        trials_results = Parallel(n_jobs=n_threads)(
            delayed(run_trial_breakout_sma_crossover)(*trial) for trial in tqdm(trails, desc="Breakout Detection (SMA crossover)"))

        # -- save results for all trials in disk
        trials_results = sorted(trials_results, key=operator.itemgetter(3), reverse=False)
        np.savetxt(trials_fname, trials_results, fmt="%d,%d,%f,%f")

    else:
        # -- ready the trials results from a previous execution
        trials_results = pd.read_csv(trials_fname, sep=',', header=None)
        trials_results = list(trials_results.to_records(index=False))

    trials_results = sorted(trials_results, key=operator.itemgetter(3), reverse=False)
    print(bcolors.OKGREEN + "Best configuration:", trials_results[0], bcolors.ENDC)

    return trials_results


# Compute the Recovery Rates for the airports considering an exponential models
def compute_recovery_rates(sma, breaks, baseline, airport_names, output_path, vis_r2_threshold=0.7):
    suptitle_font = {'size': '34', 'color': 'black', 'weight': 'bold', 'verticalalignment': 'bottom'}
    title_font = {'size': '20', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}

    axis_font = {'size': '20'}
    font_size_axis = 20
    font_size_legend = 20

    recovery_rates = []
    results = []
    for name in tqdm(airport_names, desc="Computing Recovery Rates"):
        data = baseline[name] - sma[name][breaks[name]:]
        positive_values_idx = data[data > 0].index
        Y_data = -np.log(data[positive_values_idx])
        Y_data = Y_data.values.reshape(-1, 1)

        time = sma[name][breaks[name]:][positive_values_idx].index.to_julian_date()
        # y_date = sma[name][breaks[name]:][positive_values_idx].index

        nans = np.isnan(Y_data)
        Y_data_nn = Y_data[~nans].reshape(-1, 1)
        time_nn = time[~nans.flatten()]
        # y_date = y_date[~nans.flatten()]

        model = linear_model.LinearRegression()
        model.fit(Y_data_nn, time_nn)

        time_pred = model.predict(Y_data_nn)

        recovery_rates += [model.coef_[0]]

        # print('Coefficients:' , model.coef_[0], '\t Mean squared error: %.2f' % metrics.mean_squared_error(time_nn, time_pred),               '\t R-squared: %.2f' % metrics.r2_score(time_nn, time_pred), end="\n")

        results += [[metrics.r2_score(time_nn, time_pred), metrics.mean_squared_error(time_nn, time_pred)]]

        with plt.style.context('seaborn'):
            plt.clf()
            # plt.rc('text', usetex=True)
            # plt.rc('font', family='serif')
            fig = plt.figure(figsize=(6, 5), dpi=300)
            ax = fig.add_subplot(111)

            plt.scatter(time_nn, Y_data_nn, color='blue', alpha=0.3)
            plt.plot(time_nn, Y_data_nn, color='blue', linestyle='dotted')
            plt.plot(time_pred, Y_data_nn, color='gold', linestyle='solid', linewidth=3)

            country_name = countries_iso_codes(get_airport_info(name)['iso_country'])
            sign = "+" if model.coef_[0] >= 0 else "-"
            r2_str = "R-squared ({0:.3f})".format(results[-1][0])
            rr_str = "Recovery rate {0} ({1:.3f})".format(sign, model.coef_[0])

            fig.suptitle("{} ({})".format(name, country_name), fontsize=20, fontweight='bold', ha='left', x=0.1)
            ax.set_title("{}\n{}".format(r2_str, rr_str), fontdict=title_font, loc='right')

            plt.xlabel("Time", fontdict=axis_font)
            plt.ylabel('-ln(Y - Yc)', fontdict=axis_font)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # plt.tick_params(axis='both', labelsize=font_size_axis)
            plt.tight_layout()
            safe_create_dir("{}/recovery-rates".format(output_path))
            plt.savefig("{}/recovery-rates/{}.png".format(output_path, name))
            plt.close('all')

    print(bcolors.OKGREEN+"Saving the visualizations for the estimated regression models in: {}/recovery-rates/".format(output_path), bcolors.ENDC)

    results = pd.DataFrame(results, index=airport_names, columns=["R-squared", "MSE"])
    results = results.sort_values("R-squared", ascending=False)

    results.to_csv(os.path.join(output_path, "recovery-coefficients.csv"), sep=",", float_format="%.3f", index=True,
                   index_label="Airport")
    results[results.iloc[:, 0] > vis_r2_threshold].to_csv(
        os.path.join(output_path, "recovery-coefficients-r2-greater-than-{}.csv".format(vis_r2_threshold)), sep=",",
        float_format="%.3f", index=True, index_label="Airport")

    print(bcolors.OKGREEN+"Saving a CSV file containing the recovery rate coefficients in:", os.path.join(output_path, "recovery-coefficients.csv"), bcolors.ENDC)

    with plt.style.context('seaborn'):
        plt.clf()
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        fig = plt.figure(figsize=(14, 4), dpi=300)
        ax = fig.add_subplot(111)
        ax = results.iloc[:, 0].plot(kind='bar', ax=ax)
        plt.title("Regression analysis (R-squared explanation)", fontdict=title_font)
        # for p in ax.patches:
        #     ax.annotate("{0:.2f}".format(p.get_height()), (p.get_x()-0.005, p.get_height() * 1.005))

        highlight = [0, 14, 29]
        for pos in highlight:
            # pos = df.index.get_loc(highlight)
            ax.patches[pos].set_facecolor('darkblue')

        plt.tick_params(axis='both', labelsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "recovery-coefficients.png"))
        plt.close('all')

    return np.array(recovery_rates).reshape(-1, 1), results


def clustering(recovery_rates, airport_names, output_path, show_plots=False):
    names = []
    for name in airport_names:
        names += ["{} ({})".format(name, countries_iso_codes(get_airport_info(name)['iso_country']))]
    names = np.array(names)

    for n_clusters in tqdm(range(3, 5 + 1, 2), desc="Clustering the Recovery Rate Coefficients (K-means)"):

        estimator = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30, max_iter=300, random_state=42, verbose=0)
        estimator.fit(recovery_rates)

        centroids = estimator.cluster_centers_
        estimator_labels = estimator.labels_
        cnames = np.arange(n_clusters)

        markers = ["o", "*", "x", "s", "H", "d", "v", "^", "<", ">", ".", "*", "p", "s", "H", "d", "v", "^", "<", ">"]
        colors = sns.color_palette('tab10')

        plt.clf()
        fig = plt.figure(figsize=(10, 4), dpi=100, frameon=False)
        ax = fig.add_subplot(111)
        val = 0.

        cluster_names = []
        for k in cnames:
            estimations = (estimator_labels == k)
            xdata = recovery_rates[estimations].flatten()
            indexes = np.argsort(xdata)

            mean = recovery_rates[estimations].mean()
            radius = np.abs(recovery_rates[estimations] - mean).max()
            plt.plot(recovery_rates[estimations], np.zeros(xdata.shape) + val,
                     color=colors[k], marker='o', markersize=11,
                     label=" - ".join(names[estimations][indexes]))

            cluster_names += [",".join(names[estimations][indexes].tolist())]
            # ax.add_patch(plt.Circle((mean, 0), radius, color=colors[k], alpha=0.2))

            # plt.plot(recovery_rates[estimations].flatten(), np.zeros(xdata.shape) + val, 'x', 
            #     label="-".join(names[estimations][indexes]))

        plt.title('KMeans')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # xlabels = ["{:.2f}".format(d) for d in recovery_rates.flatten()]
        # plt.xticks(recovery_rates, xlabels, rotation=90)
        ax.set_xticks([])
        ax.set_yticks([])

        # plt.legend(prop={'size': 10}, loc='upper center', frameon=True, markerscale=0.8, ncol=1)
        plt.tight_layout()
        if show_plots:
            plt.show()

        filename = os.path.join(output_path, "cluster-results-n_clusters-{}.png".format(n_clusters))
        plt.savefig(filename)
        plt.close('all')

        fname = os.path.join(output_path, "cluster-results-n_clusters-{}.txt".format(n_clusters))
        np.savetxt(fname, cluster_names, fmt="%s")

        print(bcolors.OKGREEN + "Saved in:", filename, bcolors.ENDC)


def save_plot_breaks_higher_than_thr_r2(sma, breaks, airport_names, recovery_rates, results, output_path, thr_r2=0.0,
                                        filter_by_date="", show_plots=False):

    mpl.rcParams.update({'figure.max_open_warning': 35})

    suptitle_font = {'size': '30', 'color': 'black', 'weight': 'bold', 'verticalalignment': 'bottom'}
    title_font = {'size': '14', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}

    axis_font = {'size': '18'}
    font_size_axis = 16
    font_size_legend = 14

    if filter_by_date:
        figsize = (6, 4)
        desc = "Saving Visualizations of the Analyzed Time Series (from {})".format(filter_by_date)
        out_dir = os.path.join(output_path, "visualizations-time-series-analysis-only-from-{}".format(filter_by_date))
    else:
        figsize = (15, 5)
        desc = "Saving Visualizations of the Analyzed Time Series (All historical data)"
        out_dir = os.path.join(output_path, "visualizations-time-series-analysis-all-historical-data")

    safe_create_dir(out_dir)

    for k, name in enumerate(tqdm(airport_names, desc=desc)):
        if results[results.index == name].iloc[0][0] >= thr_r2:
            with plt.style.context('seaborn'):
                plt.clf()

                fig = plt.figure(figsize=figsize, dpi=300, frameon=False)
                ax = fig.add_subplot(111)

                country_name = countries_iso_codes(get_airport_info(name)['iso_country'])
                mid = fig.subplotpars.left
                mid = (fig.subplotpars.right + fig.subplotpars.left) / 2
                fig.suptitle("{} ({})".format(name, country_name), fontsize=20, fontweight='bold', ha='center')
                # fig, ax = plt.subplots(figsize=(15, 5), dpi=300, frameon=False)

                # r2_str = "R-squared ({0:.3f})".format(results[results.index == name].iloc[0][0])
                # sign  = '+' if recovery_rates[k][0] >= 0 else '-'
                # rr_str = "Recovery Rate {0} ({1:.3f})".format(sign, recovery_rates[k][0])
                # ax.set_title("{}\n{}".format(r2_str, rr_str), fontdict=title_font, loc='left')

                if filter_by_date:
                    ax = sma[name][sma[name].index > filter_by_date].plot(color='orange', linewidth=2, label="", ax=ax)
                else:
                    ax = sma[name].plot(color='orange', linewidth=2, label="", ax=ax)

                ax.axvline(x=breaks[name], color='r', linestyle=':', linewidth=3,
                           label="Break ({})".format(datetime.strftime(breaks[name], "%Y-%m-%d")))
                ax.axhline(y=sma[name].mean(), color='g', linestyle=':', linewidth=3,
                           label="Baseline ({0:.3f})".format(sma[name].mean()))

                plt.xlabel("Date", fontdict=axis_font)
                plt.ylabel("# of airplanes", fontdict=axis_font)

                plt.tick_params(axis='both', labelsize=font_size_axis)
                plt.legend(prop={'size': font_size_legend},
                           fancybox=True, framealpha=0.8,
                           loc='lower left',
                           # bbox_to_anchor=(0., 1.02, 1., .102), loc=4, 
                           # frameon=False, ncol=2,
                           )

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                if show_plots:
                    plt.show()

                plt.savefig(os.path.join(out_dir, "{}.png".format(name)))
                plt.close('all')

    print(bcolors.OKGREEN + "Done." + bcolors.ENDC)

# Plot the detected structural breaks
def save_plot_breaks(sma, breaks, airport_names, recovery_rates, results, output_path, show_plots=False):

    mpl.rcParams.update({'figure.max_open_warning': 35})

    title_font = {'size': '20', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}
    axis_font = {'size': '18'}

    font_size_axis = 16
    font_size_legend = 16

    plt.clf()
    fig = plt.figure(figsize=(30, 50), dpi=300, frameon=False)

    for k, name in enumerate(tqdm(airport_names, desc="Generating Visualizations for the Detected Breaking Points")):

        ax = plt.subplot(15, 2, k + 1)

        country_name = countries_iso_codes(get_airport_info(name)['iso_country'])
        name_country = "{} ({})".format(name, country_name)
        r2_str = "R-squared ({0:.3f})".format(results[results.index == name].iloc[0][0])
        sign = '+' if recovery_rates[k][0] >= 0 else '-'
        rr_str = "Recovery Rate {0} ({1:.3f})".format(sign, recovery_rates[k][0])
        # ax.set_title("{} ({})".format(name, country_name), fontdict=title_font)
        ax.set_title("{0:<20}{1:<20}{2:<20}".format(name_country, r2_str, rr_str), fontdict=title_font, loc='left')

        sma[name].plot(color='blue', linewidth=2, label="")
        ax.axvline(x=breaks[name], color='r', linestyle=':', linewidth=3, label="Break ({})".format(breaks[name]))
        ax.axhline(y=sma[name].mean(), color='g', linestyle=':', linewidth=3, label="Baseline ({})".format(sma[name].mean()))
        ax.set_xlabel("Date", fontdict=axis_font)
        ax.set_ylabel("# of airplanes", fontdict=axis_font)

        ax.tick_params(axis='both', labelsize=font_size_axis)
        ax.legend(prop={'size': font_size_legend}, loc='upper left',
                  fancybox=True, framealpha=0.8,
                  )

    plt.tight_layout(pad=2)

    if show_plots:
        plt.show()
    plt.savefig(os.path.join(output_path, "breakouts-visualization.pdf"))
    plt.close('all')

    print(bcolors.OKGREEN + "Saved in:",os.path.join(output_path, "breakouts-visualization.pdf") + bcolors.ENDC)


def compute_correlations(sma, breaks, airport_names, recovery_rates, output_path):
    # Compute the correlations between the number of counted airplanes, after the break, and the 
    # new cases and new deaths of COVID-19

    recovery_rates = recovery_rates.flatten()
    correlation_results = []
    for k, name in enumerate(tqdm(airport_names, desc="Computing Correlations")):

        covid_data = get_covid_info(airport_code=name)
        covid_data = simple_moving_average_covid(covid_data, wsize=14)
        start_covid_index = covid_data.index[0]
        sma_after_break = sma[name][breaks[name]:]
        sma_after_break = sma_after_break[sma_after_break.index >= start_covid_index]

        cdata = covid_data[covid_data.index.isin(sma_after_break.index)]

        try:
            total_cases = stats.pearsonr(cdata['total_cases'], sma_after_break)[0]
            total_deaths = stats.pearsonr(cdata['total_deaths'], sma_after_break)[0]
            new_cases = stats.pearsonr(cdata['new_cases'], sma_after_break)[0]
            new_deaths = stats.pearsonr(cdata['new_deaths'], sma_after_break)[0]
            correlation_results += [
                [name, countries_iso_codes(get_airport_info(name)['iso_country']), recovery_rates[k], total_cases,
                 total_deaths, new_cases, new_deaths]]

        except Exception as e:
            raise(e)


    columns = ["Airtport's Name",
               "Country",
               "recovery_rates",
               "corr_total_cases_x_recovery_rate",
               "corr_total_deaths_x_recovery_rate",
               "corr_new_cases_x_recovery_rate",
               "corr_new_deaths_x_recovery_rate",
               ]

    correlation_results = pd.DataFrame(correlation_results, columns=columns)
    correlation_results.to_csv(os.path.join(output_path, "correlations_covid19_x_recovery_rates.csv"), sep=",",
                               float_format="%.3f", index=False)

    print(bcolors.OKGREEN + "Saved in:", os.path.join(output_path, "correlations_covid19_x_recovery_rates.csv"), bcolors.ENDC)

def read_time_series(time_series_path, show_plots):

    airport_names = []
    all_datetimes, all_X, all_Y = [], [], []

    # -- get paths to files containing the timeseries
    filenames = retrieve_filenames(time_series_path, '.log')
    filenames = [fn for fn in filenames if '_day' in fn]

    for arg in tqdm(filenames, desc="Reading Time Series"):
        X, Y, dates = [], [], []
        with open(arg, 'r') as fp:
            start = None
            for i, line in enumerate(fp):
                d, count, score = line.split()
                str_d = line.split()[0]
                if float(score) > 0.5:
                    count = float(count)
                    d = [int(x) for x in d.split('-')]
                    if len(d) == 2:
                        if d[1] == 12:
                            d[0] += 1
                            d[1] = 1
                        else:
                            d[1] += 1
                        d.append(1)
                    dates.append(datetime(d[0], d[1], d[2]))
                    d = date(d[0], d[1], d[2])
                    if start is None:
                        start = d
                    X.append((d - start).days)
                    Y.append(count)
        all_X += [X]
        all_Y += [Y]
        all_datetimes += [dates]
        airport_names += [os.path.basename(arg).split('_')[0]]

    data = []
    for name, dt, value in zip(airport_names, all_datetimes, all_Y):
        timestamps = pd.to_datetime(dt)
        data += [pd.Series(value, index=dt, name=name).resample('D').mean()]

    data = pd.concat(data, axis=1)
    data = data.interpolate('polynomial', order=1)
    data = data.transform(fill_missing_data)

    if show_plots:
        data.plot(figsize=(22, 5))
        plt.legend(ncol=5)
        plt.show()
        plt.close('all')

    return data, airport_names

def compute_breaking_points_in_time_series(data, airport_names, breakout_algo, output_path, n_threads):

    if 'twitter-algo' == breakout_algo:

        trials_results = find_parameters_breakout_detection_twitter(data, airport_names, output_path, n_threads)

        sma = simple_moving_average(data, wsize=trials_results[0][0])
        baseline = sma.mean()
        breaks = breakout_detection_twitter(data, sma, airport_names, trials_results[0][0], trials_results[0][1], trials_results[0][2])

    elif 'sma-algo' == breakout_algo:

        trials_results = find_parameters_breakout_detection_sma_crossover(data, airport_names, output_path, n_threads)

        sma = simple_moving_average(data, wsize=trials_results[0][0])
        baseline = sma.mean()
        breaks = breakout_detection_sma_crossover(data, sma, trials_results[0][0], trials_results[0][1])

    else:
        raise("Algorithm not implemented yet.")

    return sma, baseline, breaks

def main():


    # -- define the arguments available in the command line execution
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--time_series_path', default="timeseries", type=str,
                        help='The path to time series built by counting the detected flying airplanes (default=%(default)s).')
    parser.add_argument('--breakout_algo', default="sma-algo", type=str, choices=["sma-algo", "twitter-algo"],
                        help='Available algorithms to detect breakout in time series (default=%(default)s).')
    parser.add_argument('--output_path', default="working", type=str,
                        help='The path to the output directory (default=%(default)s).')
    parser.add_argument('--n_threads', default=multiprocessing.cpu_count()//2, type=int,
                        help='Number of threads can run in parallel (default=%(default)s).')
    parser.add_argument('--show_plots', action='store_true')

    args = parser.parse_args()

    time_series_path = args.time_series_path
    breakout_algo = args.breakout_algo
    output_path = args.output_path
    show_plots = args.show_plots
    n_threads = args.n_threads

    output_path = os.path.join(output_path, breakout_algo)

    # -- reading the time series
    data, airport_names = read_time_series(time_series_path, show_plots)

    # -- compute breaking points for all airports
    sma, baseline, breaks = compute_breaking_points_in_time_series(data, airport_names, breakout_algo, output_path, n_threads)

    # -- compute the recovery rate for all airports
    recovery_rates, results = compute_recovery_rates(sma, breaks, baseline, airport_names, output_path)

    save_plot_breaks(sma, breaks, airport_names, recovery_rates, results, output_path)
    save_plot_breaks_higher_than_thr_r2(sma, breaks, airport_names, recovery_rates, results, output_path)
    save_plot_breaks_higher_than_thr_r2(sma, breaks, airport_names, recovery_rates, results, output_path,
                                        filter_by_date="2020-01")

    clustering(recovery_rates, airport_names, output_path)

    compute_correlations(sma, breaks, airport_names, recovery_rates, output_path)


if __name__ == '__main__':
    main()

