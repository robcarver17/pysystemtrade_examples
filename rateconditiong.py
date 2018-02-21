

from pysystemtrade.systems.provided.futures_chapter15.basesystem import futures_system
from pysystemtrade.syscore.pdutils import pd_readcsv

from matplotlib import pyplot
import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day
from random import uniform
from scipy.stats import ttest_ind

rate_to_use = pd_readcsv("/home/rob/workspace3/pysystemtrade_examples/DFF.csv", date_index_name="DATE")
us_5_year = pd_readcsv("/home/rob/workspace3/pysystemtrade_examples/FRED-DGS5.csv", date_index_name="Date")

rate_to_use = us_5_year.Value
rate_to_use = rate_to_use.sort_index()
rate_to_use = rate_to_use.asfreq(Day()).ffill()

DAYSINYEAR = 365
rate_average = rate_to_use.rolling(int(DAYSINYEAR * 15), min_periods=DAYSINYEAR).mean()
rate_adjusted = rate_to_use / rate_average  # mean 0.79 current level 0.68
rate_raw = rate_to_use

rate_change = rate_to_use.ffill().diff().rolling(window=DAYSINYEAR).sum()
rate_change_adjusted = rate_change / rate_average

system = futures_system()
system.config.instrument_weights = dict(EDOLLAR = .25, US5 = .25, US10 = .25, US20 = .25)

# use this line for sp500
#system.config.instrument_weights = dict(SP500=1.0)

system.config.use_instrument_div_mult_estimates=True
system.config.forecast_weights = dict(ewmac16_64 = 0.2, ewmac32_128 = 0.2, ewmac64_256 = 0.2, carry=0.4)

# partition
def partition(pandlcurve, conditioning_variable, pbins=5):

    # match
    conditioning_variable_match = conditioning_variable.reindex(pandlcurve.index, method="ffill")

    # avoid look ahead
    conditioning_variable_shift = conditioning_variable_match.shift(1)

    cmin = np.nanmin(conditioning_variable_shift)
    cmax = np.nanmax(conditioning_variable_shift)
    crange = cmax - cmin

    bounds = [np.nanpercentile(conditioning_variable_shift, xpoint) for xpoint in np.arange(0, 100.001, step=100.0/(pbins))]

    pandl_list = []
    bound_names = []
    for (lower_bound, upper_bound) in zip(bounds[:-1], bounds[1:]):
        pandl = pandlcurve[(conditioning_variable_shift>lower_bound) & (conditioning_variable_shift<=upper_bound)]

        pandl_list.append(pandl)

        bound_name = "%.2f to %.2f (SR:%.2f)" % (lower_bound, upper_bound, pandl.mean()*16/pandl.std())
        bound_names.append(bound_name)

    return (pandl_list, bound_names)


data = [rate_raw, rate_adjusted, rate_change, rate_change_adjusted]


def partitionit(system, data, predictor_name , instrument_name, cond_name , pbins=2):

    rate_raw, rate_adjusted, rate_change, rate_change_adjusted = data

    if predictor_name =="ALL":
        if instrument_name == "ALL":
            # all predictors, all instruments
            pandl_curve = system.accounts.portfolio().percent()
        else:
            # all predictors, one instrument
            pandl_curve = system.accounts.pandl_for_instrument(instrument_name).percent()

    elif predictor_name == "LONG_ONLY":
        if instrument_name =="ALL":
            pandl_curves = [system.rawdata.daily_returns(instrument_code) / system.rawdata.daily_denominator_price(
                instrument_code) for instrument_code in system.get_instrument_list()]

            # normalise individual vol
            pandl_curves = [pandl / pandl.std() for pandl in pandl_curves]

            # combine
            pandl_curve = pd.concat(pandl_curves, axis=1)
            pandl_curve = pandl_curve.sum(axis=1)

        else:
            # single instrument, long only
            pandl_curve = 100*system.rawdata.daily_returns(instrument_name)/system.rawdata.daily_denominator_price(instrument_name)

    else:
        if instrument_name == "ALL":
            # individual predictors, all instruments

            pandl_curve = system.accounts.pandl_for_trading_rule_weighted(predictor_name).percent()
        else:
            # individual predictor, one instrument
            pandl_curve = system.accounts.pandl_for_instrument_forecast(instrument_name, predictor_name).percent()

    pandl_curve = pandl_curve[abs(pandl_curve) < 10]

    if cond_name == "Raw":
        conditioning_variable = rate_raw
    elif cond_name == "Adjusted":
        conditioning_variable = rate_adjusted
    elif cond_name =="Change":
        conditioning_variable = rate_change
    elif cond_name =="Adjusted Change":
        conditioning_variable = rate_change_adjusted
    else:
        raise Exception("%s not valid" % cond_name)

    pandl_list, bounds = partition(pandl_curve, conditioning_variable, pbins)

    return pandl_curve, pandl_list, bounds, conditioning_variable


def box_whisker_plot(system, data, predictor_name , instrument_name, cond_name , pbins=2):
    pandl_curve, pandl_list, bounds, conditioning_variable = partitionit(system, data, predictor_name , instrument_name, cond_name , pbins)

    pyplot.boxplot(pandl_list)
    locs, labels = pyplot.xticks()
    pyplot.xticks(locs, bounds)
    pyplot.title("%s %s, Interest Rate %s, Current value %.2f" % (predictor_name, instrument_name, cond_name, conditioning_variable.values[-1]))

# bootstrap sampling distributions

def SR_sample(sample_pandl):
    # business days
    return (DAYSINYEAR**.5)*sample_pandl.mean()/sample_pandl.std()

def generate_random_index(boot_length):
    random_index = [int(uniform(0, boot_length)) for notUsed in range(boot_length)]
    return random_index

def sampling_distribution_SR(single_pandl, boot_count = 100):

    boot_length = len(single_pandl.index)
    list_of_indices = [generate_random_index(boot_length) for notUsed in range(boot_count)]
    list_of_random_sample_periods = [single_pandl[single_index] for single_index in list_of_indices]

    SR_samples = [SR_sample(sample_period) for sample_period in list_of_random_sample_periods]

    return SR_samples


def t_tests(pandl_list):
    SR_list = [SR_sample(sample_pandl) for sample_pandl in pandl_list]

    lowest_SR = SR_list.index(min(SR_list))

    target_pandl = pandl_list[lowest_SR]
    t_list = []
    for compare_pandl in pandl_list:
        t_list.append(ttest_ind(target_pandl, compare_pandl))

    return t_list

def multiple_hist_plot(pandl_list, bounds, boot_count = 100):

    SR_distributions = [sampling_distribution_SR(single_pandl, boot_count) for single_pandl in pandl_list]

    bins = min(int(boot_count/50), 10)

    for SR_dist in SR_distributions:
        pyplot.hist(SR_dist, bins=bins)

    t_results = t_tests(pandl_list)
    plot_labels = ["%s p: %.3f" % (bound_label, test_result.pvalue)
                   for (bound_label, test_result) in zip(bounds, t_results)]

    pyplot.legend(plot_labels)

pandl_curve, pandl_list, bounds, conditioning_variable = partitionit(system, data, "ALL", "EDOLLAR", "Adjusted Change", pbins=2)


multiple_hist_plot(pandl_list, bounds, boot_count = 2000)