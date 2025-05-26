def __find_max_min(lst, pp=99, perc=0):

    from numpy import nanpercentile

    maxs, mins = [], []

    for l in lst:
        maxs.append(nanpercentile(l, pp))
        mins.append(nanpercentile(l, 100-pp))

    if perc == 0:
        return min(mins), max(maxs)
    else:
        _min = min(mins)
        _max = max(maxs)
        xx = _max*(1+perc) -_max

        return  _min-xx, _max+xx