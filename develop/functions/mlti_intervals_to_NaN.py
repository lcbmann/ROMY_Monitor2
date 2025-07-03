def __mlti_intervals_to_NaN(df_in, key_fs, key_time, mlti_t1, mlti_t2, t_offset_sec=120):

    from numpy import nan, where, full
    from tqdm.notebook import tqdm

    times = df_in[key_time]

    mask = full((len(times)), True, dtype=bool)

    df_in[key_fs+'_nan'] = df_in[key_fs]

    idx = 0
    for nn, tt in enumerate(times):

        if idx >= len(mlti_t1):
            continue
        else:
            t1, t2 = (mlti_t1[idx]-t_offset_sec), (mlti_t2[idx]+t_offset_sec)

        if tt >= t1:
            mask[nn] = False
        if tt > t2:
            idx += 1

    df_in[key_fs+'_nan'].where(mask, other=nan, inplace=True)

    return df_in