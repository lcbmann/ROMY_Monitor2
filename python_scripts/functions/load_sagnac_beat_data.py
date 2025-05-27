def __load_sagnac_beat_data(tbeg, tend, ring, path_to_data):

    from datetime import date
    from pandas import read_pickle, concat, DataFrame, date_range, merge
    from obspy import UTCDateTime


    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    dd1 = date.fromisoformat(str(tbeg.date))
    dd2 = date.fromisoformat(str(tend.date))

    # generate dates with 1 minute samples
    dates = date_range(dd1, dd2, freq="1min")

    # prepare data frame
    df0 = DataFrame()

    # prepare dummy data frame
    df_dummy = DataFrame()

    # add column for merging
    df_dummy["dt"] = [str(UTCDateTime(_d))[:16] for _d in dates]

    # add column to replace nan from merging with proper dates
    df_dummy["dummy_times"] = [UTCDateTime(_d) for _d in dates]


    for dat in date_range(dd1, dd2):

        # build file name
        file = f"{str(dat)[:4]}/R{ring}/FJ{ring}_"+str(dat)[:10].replace("-", "")+".pkl"

        try:
            df00 = read_pickle(path_to_data+file)
            df0 = concat([df0, df00])
        except:
            print(f"error for {file}")

    if df0.empty:
        print(" -> empty dataframe!")
        return df0

    # add column for merging
    df0["dt"] = [str(_dt)[:16] for _dt in df0.times_utc]

    try:
        df = merge(left=df_dummy, right=df0, on="dt", how="outer")
    except:
        print("-> no merge")

    # trim to given start and end date
    df = df[(df.dt >= tbeg) & (df.dt < tend)]

    # fill possible missing times_utc that turned to nan in merge with prepared dummy times
    df['times_utc'] = df['times_utc'].fillna(df['dummy_times'])

    # convert to UTCDateTime objects
    df['times_utc'] = [UTCDateTime(_t) for _t in df['times_utc']]

    # trim to defined times
    # df = df[(df.times_utc >= tbeg) & (df.times_utc < tend)]

    # correct seconds
    df['times_utc_sec'] = [abs(tbeg - UTCDateTime(_t)) for _t in df['times_utc']]

    # remove helper columns
    df.drop(columns=["dt", "dummy_times"], inplace=True)

    return df

#     from obspy import UTCDateTime
#     from datetime import date
#     from pandas import read_pickle, concat, DataFrame, date_range

#     t1 = date.fromisoformat(str(UTCDateTime(tbeg).date))
#     t2 = date.fromisoformat(str((UTCDateTime(tend)-86400).date))

#     df = DataFrame()
#     for dat in date_range(t1, t2):
#         print(dat)
#         file = f"FJ{ring}_"+str(dat)[:10].replace("-", "")+".pkl"
#         try:
#             df0 = read_pickle(path_to_data+file)
#             df = concat([df, df0])
#         except:
#             print(f"error for {file}")

#     df.reset_index(inplace=True)

#     return df
