def __get_fft(signal_in, dt, window=None):

    '''
    Calculating a simple 1D FastFourierSpectrum of a time series.

    RETURN:

    frequencies, spectrum, phase

    TEST:

    >>> spectrum, frequencies, phase = __fft(signal_in, dt ,window=None,normalize=None)
    '''

    from scipy.fft import fft, fftfreq, fftshift
    from scipy import signal
    from numpy import angle, imag

    # determine length of the input time series
    n = int(len(signal_in))

    # calculate spectrum (with or without window function applied to time series)
    if window:
        win = signal.get_window(window, n);
        spectrum = fft(signal_in * win)

    else:
        spectrum = fft(signal_in)

    # calculate frequency array
    frequencies = fftfreq(n, d=dt)

    # correct amplitudes of spectrum
    magnitude = abs(spectrum) * 2.0 / n

    phase = angle(spectrum, deg=False)
    # phase = imag(spectrum)

    # return the positive frequencies
    return frequencies[0:n//2], magnitude[0:n//2], phase[0:n//2]