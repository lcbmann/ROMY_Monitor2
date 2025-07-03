def __load_beam_wander_data(tbeg, tend, path_to_data, cam, verbose=False):

    from obspy import UTCDateTime
    from datetime import date
    from pandas import read_pickle, concat, DataFrame, date_range, to_datetime

    t1 = date.fromisoformat(str(UTCDateTime(tbeg).date))
    t2 = date.fromisoformat(str((UTCDateTime(tend)).date))

    errors = []

    df = DataFrame()
    for dat in date_range(t1, t2, inclusive="left"):

#        file = f"data{cam}/{str(dat)[:10].replace('-','')}.pkl"
        file = f"{str(dat)[:10].replace('-','')}.pkl"
        try:
            if verbose:
                print("loading:", path_to_data+file)
            df0 = read_pickle(path_to_data+file)
            df = concat([df, df0])
        except:
            errors.append(file)
            if verbose:
                print(f" -> error for {file}")


    # remove NaN from time column
    try:
        df.dropna(subset=['time'], inplace=True)
    except:
        pass

    if verbose:
        print(df.head())

    # reset the index column
    df.reset_index(inplace=True, drop=True)

    # add column for relative time in seconds
    try:
        df['time_sec'] = [UTCDateTime(_t) - UTCDateTime(df.time.iloc[0]) for _t in df.time]
    except:
        df['time_sec'] = range(0, len(df))

    if len(errors) > 0:
        print(f"errors {cam}: ", errors)

    return df