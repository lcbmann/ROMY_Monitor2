def __load_furt_stream(starttime, endtime, show_raw=False, sampling_rate=1.0, path_to_archive = '/bay200/gif_online/FURT/WETTER/'):

    '''
    Load a selection of data of FURT weather station for certain times and return an obspy stream


    PARAMETERS:
        - config:    configuration dictionary
        - show_raw:  bool (True/False) -> shows raw data FURT head


    RETURN:
        - stream

    EXAMPLE:
    >>> __load_furt_stream(config, show_raw=False, path_to_archive = '/bay200/gif_online/FURT/WETTER/')

    '''

    from pathlib import Path
    from obspy import UTCDateTime
    from tqdm.notebook import tqdm_notebook
    from numpy import arange, ones, nan
    from obspy import Stream
    from pandas import concat, to_datetime, read_csv, DataFrame

    def __add_trace(cha, tbeg, dat, dt=1):

        from obspy import Trace, UTCDateTime
        from numpy import array

        tr = Trace()
        tr.stats.station = 'FURT'
        tr.stats.network = 'BW'
        tr.stats.channel = str(cha)
        tr.stats.sampling_rate = 1/dt
        tr.stats.starttime = UTCDateTime(tbeg)
        tr.data = array(dat)

        return tr

    def __resample(df, freq='1S'):

        # check for NaN in dates
        if df.date.isna().any():
            print(" -> NaN values found and removed from column date")
            df = df.dropna(axis=0, subset=["date"])
            try:
                df["date"] = df["date"].astype(int)
            except:
                df["date"] = df["Date"].astype(int)

        # make column with datetime
        df['datetime'] = df['date'].astype(str).str.rjust(6,"0")+" "+df['time'].astype(str).str.rjust(6,"0")

        # drop datetime duplicates
        df = df[df.duplicated("datetime", keep="first") != True]

        # convert to pandas datetime object
        # df['datetime'] = to_datetime(df['datetime'], format="%d%m%y %H%M%S", errors="coerce")
        df['datetime'] = to_datetime(df['datetime'], format="%d%m%y %H%M%S", errors="ignore")

        # set datetime column as index
        df.set_index('datetime', inplace=True)

        # remove duplicates
        df = df[~df.index.duplicated()]

        # resample
        df = df.asfreq(freq=freq)

        return df

    starttime = UTCDateTime(starttime)
    endtime = UTCDateTime(endtime)

    output_text = []

    new_delta = 1/sampling_rate

    if not Path(path_to_archive).exists():
        output_text.append(f"  -> Path: {path_to_archive}, does not exist!")
        print(f"  -> Path: {path_to_archive}, does not exists!")

    # declare empyt dataframe
    df = DataFrame()

    for i, date in enumerate(arange(starttime.date, (endtime+86400+10).date)):

        date = UTCDateTime(str(date)).date
        filename = f'FURT.WSX.D.{str(date.day).rjust(2,"0")}{str(date.month).rjust(2,"0")}{str(date.year).rjust(2,"0")[-2:]}.0000'

        if show_raw:
            df0 = read_csv(path_to_archive+filename)
            print(df0.columns.tolist())

        try:
            try:
                try:
                    df0 = read_csv(path_to_archive+filename, header=0, usecols=[0,1,5,8,10,12,13,14], names=['date', 'time', 'Dm', 'Sm', 'T', 'H', 'P','Rc'])
                except:
                    df0 = read_csv(path_to_archive+filename, usecols=[0,1,5,8,10,12,13,14], names=['date', 'time', 'Dm', 'Sm', 'T', 'H', 'P','Rc'])
            except:
                print(f" -> loading of {filename} failed!")

            # substitute strings with floats

            # __________________________________________
            # air temperature Ta in degree C
            # try:
                # df0['T']  = [float(str(str(t).split("=")[1]).split("C")[0]) for t in df0['T']]
            # except:
            #     df0['T'] = ones(len(df0['T']))*nan
            #     print(f" -> {filename}: subsituted T with nan...")

            TT = ones(len(df0['T']))*nan
            for _n, t in enumerate(df0['T']):
                try:
                    TT[_n] = float(str(str(t).split("=")[1]).split("C")[0])
                except:
                    # print(t)
                    continue
            df0['T'] = TT

            # __________________________________________
            # air pressure Pa in hPa

            PP = ones(len(df0['P']))*nan
            for _n, p in enumerate(df0['P']):
                try:
                    PP[_n] = float(str(str(p).split("=")[1]).split("H")[0])
                except:
                    # print(p)
                    continue
            df0['P'] = PP

            # __________________________________________
            # ## relative humiditiy Ua in %RH

            HH = ones(len(df0['H']))*nan
            for _n, h in enumerate(df0['H']):
                try:
                    HH[_n] = float(str(str(h).split("=")[1]).split("P")[0])
                except:
                    # print(h)
                    continue
            df0['H'] = HH

            # __________________________________________
            # ## rain accumulation in mm

            Rc = ones(len(df0['Rc']))*nan
            for _n, rc in enumerate(df0['Rc']):
                try:
                    Rc[_n] = float(str(str(rc).split("=")[1]).split("M")[0])
                except:
                    # print(rc)
                    continue
            df0['Rc'] = Rc

            # __________________________________________
            # ## wind speed average in m/s

            Sm = ones(len(df0['Sm']))*nan
            for _n, sm in enumerate(df0['Sm']):
                try:
                    Sm[_n] = float(str(str(sm).split("=")[1]).split("M")[0])
                except:
                    # print(sm)
                    continue
            df0['Sm'] = Sm

            # __________________________________________
            # ## wind direction average in degrees

            Dm = ones(len(df0['Dm']))*nan
            for _n, dm in enumerate(df0['Dm']):
                try:
                    Dm[_n] = float(str(str(dm).split("=")[1]).split("D")[0])
                except:
                    # print(dm)
                    continue
            df0['Dm'] = Dm

            # __________________________________________

            # replace error indicating values (-9999, 999.9) with NaN values
#             df0.replace(to_replace=-9999, value=nan, inplace=True)
#             df0.replace(to_replace=999.9, value=nan, inplace=True)

            if df.empty:
                df = df0
            else:
                try:
                    df = concat([df, df0])
                except:
                    print(f"  -> failed to concat for {filename}")
        except Exception as e:
            print(e)
            output_text.append(f"  -> {filename}, failed!")
#             print(f"  -> File: {filename}, does not exists!")

    # reset the index for the joined frame
    df.reset_index(inplace=True, drop=True)

    # resample dataframe and avoid data gaps
    try:
        df = __resample(df, freq=f'{new_delta}S')
    except Exception as e:
        print(e)

    for text in output_text:
        print(text)

    df_starttime = UTCDateTime(df.index[0])

    # create stream and attach traces
    st0 = Stream()
    st0 += __add_trace("LAT", df_starttime, df['T'], dt=new_delta)
    st0 += __add_trace("LAP", df_starttime, df['P'], dt=new_delta)
    st0 += __add_trace("LAH", df_starttime, df['H'], dt=new_delta)
    st0 += __add_trace("LAR", df_starttime, df['Rc'], dt=new_delta)
    st0 += __add_trace("LAW", df_starttime, df['Sm'], dt=new_delta)
    st0 += __add_trace("LAD", df_starttime, df['Dm'], dt=new_delta)

    # correct mseed naming
    # st0 += __add_trace("LKO", df_starttime, df['T'], dt=new_delta)
    # st0 += __add_trace("LDO", df_starttime, df['P'], dt=new_delta)
    # st0 += __add_trace("LIO", df_starttime, df['H'], dt=new_delta)
    # st0 += __add_trace("LXR", df_starttime, df['Rc'], dt=new_delta)

    # trim to specfied time period
    st0.trim(starttime, endtime-new_delta/2)

    t1, t2 = endtime-new_delta, st0.select(channel='*T')[0].stats.endtime
    if t1 != t2:
        print(f"Specified end: {t1} \nTrace end:     {t2}")

    return st0

# END OF FILE





