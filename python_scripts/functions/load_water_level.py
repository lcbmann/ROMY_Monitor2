def __load_water_level(tbeg, tend, path_to_data="/lamont/Pegel/"):

    from datetime import date
    from pandas import read_csv, concat, DataFrame, date_range
    from obspy import UTCDateTime

    # path_to_data = "/lamont/Pegel/"

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    if tbeg < UTCDateTime("2023-11-26"):
        print(f" -> no good data before 2023-11-26!")
        tbeg = UTCDateTime("2023-11-27")

    dd1 = date.fromisoformat(str(tbeg.date))
    dd2 = date.fromisoformat(str(tend.date))

    df = DataFrame()
    for dat in date_range(dd1, dd2):
        file = f"{str(dat)[:4]}/PG"+str(dat)[:10].replace("-", "")+".dat"
        try:
            # load data
            df0 = read_csv(path_to_data+file, delimiter=" ")

            # correct seconds
            if "hour" in df0.keys():
                df0['times_utc'] = [UTCDateTime(f"{_d[-4:]+_d[3:5]+_d[:2]} {_t}")  for _d, _t in zip(df0['day'], df0['hour'])]
                df0['times_utc_sec'] = [abs(tbeg - UTCDateTime(_t))  for _t in df0['times_utc']]
            elif "time" in df0.keys():
                df0['times_utc'] = [UTCDateTime(f"{_d[-4:]+_d[3:5]+_d[:2]} {_t}")  for _d, _t in zip(df0['day'], df0['time'])]
                df0['times_utc_sec'] = [abs(tbeg - UTCDateTime(_t))  for _t in df0['times_utc']]

            # merge
            df = concat([df, df0])

        except:
            print(df0)
            print(f"error for {file}")

    # convert data
    # to meter
    df['pegel'] = df.pegel*0.75
    # to degree celcius
    df['temperatur'] = df.temperatur*5


    # remove columns hour and day
    # if "hour" in df0.keys():
    #     df.drop(columns=["hour", "day"], inplace=True)
    # elif "time" in df0.keys():
    #     df.drop(columns=["time", "day"], inplace=True)


    # reset index to make it continous
    df.reset_index(inplace=True)

    if df.empty:
        print(" -> empty dataframe!")
        return df

    # trim to defined times
    df = df[(df.times_utc >= tbeg) & (df.times_utc < tend)]
    return df