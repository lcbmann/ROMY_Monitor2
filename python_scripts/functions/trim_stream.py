def trim_stream(st0, set_common=True, set_interpolate=False):
    '''
    Trim a stream to common start and end times (and interpolate to common times)
    '''

    from numpy import interp, arange

    def __get_size(st0):
        return [tr.stats.npts for tr in st0]

    # get size of traces
    n_samples = __get_size(st0)

    # check if all traces have same amount of samples
    if not all(x == n_samples[0] for x in n_samples):
        print(f" -> stream size inconsistent: {n_samples}")

        # if difference not larger than one -> adjust
        if any([abs(x-n_samples[0]) > 1 for x in n_samples]):

            # set to common minimum interval
            if set_common:
                _tbeg = max([tr.stats.starttime for tr in st0])
                _tend = min([tr.stats.endtime for tr in st0])
                st0 = st0.trim(_tbeg, _tend, nearest_sample=True)
                print(f"  -> adjusted: {__get_size(st0)}")

                if set_interpolate:
                    _times = arange(0, min(__get_size(st0)), st[0].stats.delta)
                    for tr in st0:
                        tr.data = interp(_times, tr.times(reftime=_tbeg), tr.data)
        else:
            # adjust for difference of one sample
            for tr in st0:
                tr.data = tr.data[:min(n_samples)]
            print(f"  -> adjusted: {__get_size(st0)}")

    return st0
