import numpy

class Settings:
    # physical variables
    use_difraction = bool()
    wave_length = numpy.float64()
    pulse_length = numpy.float64()
    phase = numpy.float64()
    chirp = numpy.float64()
    x_curvature = numpy.float64()
    intensity = numpy.float64()
    x_width = numpy.float64()
    m = numpy.float64()
    n = numpy.float64()

    evenOrders = bool()
    N0 = numpy.float64()
    a = numpy.float64()
    b = numpy.float64()

    wv = numpy.float64()
    tv = numpy.float64()
    n2e = numpy.float64()
    n2ev = numpy.float64()
    hiX11 = numpy.float64()
    hiX12 = numpy.float64()
    use_cubic = bool()
    use_raman = bool()
    use_square = bool()
    use_ion = bool()
    use_cubic_envelope = bool()
    use_fft_for_nonlinear = bool()

    # computation variables
    time_size = numpy.int32()
    space_size = numpy.int32()
    time_limit = numpy.float64()
    space_limit = numpy.float64()
    z = numpy.float64()
    uniformSave = bool()
    dz_save = numpy.float64()

    # strategy variables
    z_strategy = ""
    dz = numpy.float64()
    layer_number = numpy.int32()
    possible_iteration_number = numpy.int32()
    possible_error = numpy.float64()

    exponentSpace = bool()
    x0 = numpy.float64()

    uniformGrid = bool()
    pulse_shift = numpy.float64()

    @staticmethod
    def toFemtoseconds():
        Settings.pulse_length = Settings.pulse_length * 1e+15
        Settings.a = Settings.a * (1e+15)**3
        Settings.b = Settings.b / 1e+15
