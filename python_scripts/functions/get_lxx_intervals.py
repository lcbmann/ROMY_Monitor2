def __get_lxx_intervals(lxx_times, time_delta=60):

    from obspy import UTCDateTime
    from numpy import array

    if len(lxx_times) == 0:
        return array([]), array([])

    t1, t2 = [], []
    for k, _t in enumerate(lxx_times):

        _t = UTCDateTime(_t)

        if k == 0:
            _tlast = _t
            t1.append(UTCDateTime(str(_t)[:16]))

        if _t -_tlast > time_delta:
            t2.append(UTCDateTime(str(_tlast)[:16])+60)
            t1.append(UTCDateTime(str(_t)[:16]))

        _tlast = _t

    t2.append(UTCDateTime(str(_t)[:16])+60)

    return array(t1), array(t2)