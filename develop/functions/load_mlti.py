def __load_mlti(tbeg, tend, ring, path_to_archive):

    from obspy import UTCDateTime
    from pandas import read_csv, concat

    tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    rings = {"U":"03", "Z":"01", "V":"02", "W":"04"}

    if tbeg.year == tend.year:
        year = tbeg.year

        path_to_mlti = path_to_archive+f"romy_archive/{year}/BW/CROMY/{year}_romy_{rings[ring]}_mlti.log"

        mlti = read_csv(path_to_mlti, names=["time_utc","Action","ERROR"])

    else:

        path_to_mlti1 = path_to_archive+f"romy_archive/{tbeg.year}/BW/CROMY/{tbeg.year}_romy_{rings[ring]}_mlti.log"
        mlti1 = read_csv(path_to_mlti1, names=["time_utc","Action","ERROR"])

        path_to_mlti2 = path_to_archive+f"romy_archive/{tend.year}/BW/CROMY/{tend.year}_romy_{rings[ring]}_mlti.log"
        mlti2 = read_csv(path_to_mlti2, names=["time_utc","Action","ERROR"])

        mlti = concat([mlti1, mlti2])

    mlti = mlti[(mlti.time_utc > tbeg) & (mlti.time_utc < tend)]

    return mlti