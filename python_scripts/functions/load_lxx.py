def __load_lxx(tbeg, tend, path_to_archive):

    from obspy import UTCDateTime
    from pandas import read_csv, concat

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    if tbeg.year == tend.year:
        year = tbeg.year

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{year}/logfiles/LXX_maintenance.log"

        lxx = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

    else:

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{tbeg.year}/logfiles/LXX_maintenance.log"
        lxx1 = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

        path_to_lx_maintenance = path_to_archive+f"romy_autodata/{tend.year}/logfiles/LXX_maintenance.log"
        lxx2 = read_csv(path_to_lx_maintenance, names=["datetime","WS1","WS4","WS5","WS6","WS7","WS8","WS9","sum_all"])

        lxx = concat([lxx1, lxx2])

    lxx = lxx[(lxx.datetime > tbeg) & (lxx.datetime < tend)]

    return lxx