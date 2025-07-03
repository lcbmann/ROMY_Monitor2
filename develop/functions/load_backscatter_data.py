def __load_backscatter_data(tbeg, tend, ring, path_to_data):

    from os.path import isfile
    from obspy import UTCDateTime
    from datetime import date
    from pandas import read_pickle, concat, DataFrame, date_range

    t1 = date.fromisoformat(str(UTCDateTime(tbeg).date))
    t2 = date.fromisoformat(str((UTCDateTime(tend)-86400).date))

    df = DataFrame()
    for dat in date_range(t1, t2):
        # print(dat)
        dat_str = str(dat)[:10].replace("-", "")
        file = f"FJ{ring}_{dat_str}_backscatter.pkl"

        if not isfile(path_to_data+file):
            _path = path_to_data
            # _path = path_to_data+"sagnac_frequency/data/"

            out = DataFrame()

            filename = f"FJ{ring}_{dat_str}_backscatter.pkl"
            try:
                _df = read_pickle(_path+filename)
                out = concat([out, _df])
            except:
                print(f" -> failed: {_path}{filename}")
                continue

            if not out.empty:
                print(f" -> write to: {_path}backscatter/FJ{ring}_{dat_str}_backscatter.pkl")
                out.to_pickle(f"{_path}backscatter/FJ{ring}_{dat_str}_backscatter.pkl")
            else:
                continue

        try:
            df0 = read_pickle(path_to_data+file)
            df = concat([df, df0])
        except:
            print(f"error for {file}")

    ## trim to time interval
    df = df[df.time1 >= tbeg]
    df = df[df.time2 <= tend]

    df.reset_index(inplace=True)

    return df