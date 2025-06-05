def __get_mean_rmy_pressure(stations, t1, t2, path_to_data, plot=False):

    import matplotlib.pyplot as plt
    from andbro__read_sds import __read_sds
    from obspy import Stream
    from functions.interpolate_nan import __interpolate_nan
    from functions.smoothing import __smooth
    from obspy.signal.cross_correlation import correlate, xcorr_max
    from numpy import arange, roll, array

    ps0 = Stream()

    for jj in stations:
        if jj == "PROMY":
            loc, cha = "03", "LDI"
        else:
            loc, cha = "", "LDO"

        ps0 += __read_sds(path_to_data+"temp_archive/", f"BW.{jj}.{loc}.{cha}", t1-10, t2+10)

    ps0 = ps0.trim(t1, t2)

    # update stations
    stations = []
    for tr in ps0:
        stations.append(tr.stats.station)

        # smooth data
        tr.data = __smooth(tr.data, 30)

    # convert from hPa to Pa
    for tr in ps0:
        if "03" in tr.stats.location:
            tr.data = __interpolate_nan(tr.data)
        else:
            tr.data = tr.data * 100 # scale to Pa
            tr.data = __interpolate_nan(tr.data)

    # equalize number of samples
    npts_min = ps0[0].stats.npts
    for tr in ps0:
        if tr.stats.npts < npts_min:
            npts_min
    for tr in ps0:
        diff = abs(tr.stats.npts - npts_min)
        if diff != 0:
            tr.data = tr.data[:-diff]

    print(ps0)

    ps0 = ps0.detrend("linear")
    ps0 = ps0.detrend("simple")
    ps0 = ps0.filter("highpass", freq=1/(6*3600), corners=4, zerophase=True);

    # shift traces to compute mean of array
    shifted = []

    for i, sta in enumerate(stations):

        if i == 0:
            arr0 = ps0.select(station=sta)[0].data
            shifted.append(arr0)
            continue
        else:
            arr1 = ps0.select(station=sta)[0].data

        Nshift = len(arr0)

        dt = ps0[0].stats.delta

        ccf1 = correlate(arr0, arr1, shift=Nshift, demean=False, normalize='naive', method='fft')

        cclags = arange(-Nshift, Nshift+1) * dt

        shift1, value1 = xcorr_max(ccf1)

        arr1_shifted = roll(arr1, shift1)

        shifted.append(arr1_shifted)

        # compute mean
        _mean = array([])
        for i, arr in enumerate(shifted):
            if i == 0:
                _mean = arr
            else:
                _mean = _mean + arr

    mean = ps0[0].copy()
    mean.stats.station = "RMY"
    mean.stats.location = "00"
    mean.stats.channel = "LDO"
    mean.data = _mean/(i+1)

    # checkup plot
    if plot:
        times = ps0[0].times()/3600
        fig = plt.figure(figsize=(15, 5))
        for i, x in enumerate(shifted):
            plt.plot(times, x, label=stations[i], zorder=2)
        plt.plot(times, mean.data, "k", zorder=2)
        plt.legend()
        plt.grid(ls="--", color="grey", alpha=0.4)
        plt.ylabel("Pressure (Pa)", fontsize=12)
        plt.xlabel("Time (hour)", fontsize=12)
        plt.show();

    if plot:
        return mean, fig
    else:
        return mean