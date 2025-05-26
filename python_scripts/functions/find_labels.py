def __find_lables(_df, _key, t1, t2, nth=1):

    from pandas import date_range
    from obspy import UTCDateTime

    dates = date_range(UTCDateTime(t1).date, UTCDateTime(t2).date)

    lbl_times, lbl_index = [], []
    for _d in dates:
        try:
            _tmp = _df[_df[_key].astype(str).str.contains(f"{_d.isoformat()[0:10]}T00:", na=False)].min()
            _idx = _df[_df[_key] == _tmp[_key]].index[0]
            lbl_times.append(_tmp[_key])
            lbl_index.append(_idx)
        except Exception as e:
            print(f" -> failed for {_d}")
            print(e)

    return lbl_times[::nth], lbl_index[::nth]