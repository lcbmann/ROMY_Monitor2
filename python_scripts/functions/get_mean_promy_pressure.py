def __get_mean_promy_pressure(stations, t1, t2, path_to_data, plot=False):

    import matplotlib.pyplot as plt
    from andbro__read_sds import __read_sds
    from numpy import array
    from obspy import Stream
    from functions.interpolate_nan import __interpolate_nan
    from functions.smoothing import __smooth

    ps0 = Stream()

    for jj in stations:
        ps0 += __read_sds(path_to_data+"temp_archive/", f"BW.WROMY.{jj}.LDI", t1, t2)

    # convert from hPa to Pa
    for tr in ps0:
        print(tr)
        if "03" in tr.stats.location:
            continue
        else:
            tr.data = tr.data * 100 # scale to Pa

        # smooth data
        tr.data = __smooth(tr.data, 30)

        # remove NaN values
        tr.data = __interpolate_nan(tr.data)


    # compute mean
    _mean = array([])
    for i, tr in enumerate(ps0):
        if i == 0:
            _mean = tr.data
        else:
            _mean = _mean + tr.data


    mean = ps0[0].copy()
    mean.stats.location = "00"
    mean.stats.station = "PROMY"
    mean.data = _mean/(i+1)

    # checkup plot
    if plot:
        plt.figure(figsize=(15, 5))
        for i, x in enumerate(ps0):
            # plt.figure(figsize=(15, 5))
            plt.plot(x.data, label=stations[i])
        plt.plot(mean.data, "k")
        plt.legend()
        plt.ylabel("Pressure (Pa)")
        plt.xlabel("Samples")
        plt.show();

    return mean