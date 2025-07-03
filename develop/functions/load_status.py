def __load_status(tbeg, tend, ring, path_to_data):

    from datetime import date
    from pandas import read_pickle, concat, DataFrame, date_range
    from obspy import UTCDateTime
    from os.path import isfile
    from numpy import array, arange, ones, nan

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    dd1 = date.fromisoformat(str(tbeg.date))
    dd2 = date.fromisoformat(str(tend.date))

    ## dummy
    def __make_dummy(date):
        NN = 1440
        df_dummy = DataFrame()
        df_dummy['times_utc'] = array([UTCDateTime(date) + _t for _t in arange(30, 86400, 60)])
        for col in ["quality", "fsagnac", "mlti", "ac_threshold", "dc_threshold"]:
            df_dummy[col] = ones(NN)*nan

        return df_dummy

    df = DataFrame()

    missing, error = 0, 0

    for dat in date_range(dd1, dd2):
        file = f"{str(dat)[:4]}/BW/R{ring}/R{ring}_"+str(dat)[:10]+"_status.pkl"

        try:

            if not isfile(f"{path_to_data}{file}"):
                missing += 1
                # print(f" -> no such file: {file}")
                df = concat([df, __make_dummy(dat)])
            else:
                df0 = read_pickle(path_to_data+file)
                df = concat([df, df0])
        except Exception as e:
            print(e)
            error += 1
            # print(f" -> error for {file}")

    if df.empty:
        print(" -> empty dataframe!")
        return df

    ## trim to defined times
    df = df[(df.times_utc >= tbeg) & (df.times_utc < tend)]

    ## correct seconds
    df['times_utc_sec'] = [abs(tbeg - UTCDateTime(_t))  for _t in df['times_utc']]

    print(f" -> {missing} missing files")
    print(f" -> {error} errors occurred")

    return df