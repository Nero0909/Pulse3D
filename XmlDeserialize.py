__author__ = 'Nero'

from lxml import etree
import xmltodict
import numpy

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class Deserializer:
    def __init__(self, xml_path):
        self.d = self.xmlToDic(xml_path)

    def xmlToDic(self, xml_path):
        tree = etree.parse(xml_path)
        xml_str = etree.tostring(tree, encoding='utf8', method='xml')
        return xmltodict.parse(xml_str)

    def deserialize(self, settings):
        # physical variables
        settings.use_difraction = str2bool(self.d['project']['physical']['@use-difraction'])
        settings.wave_length = numpy.float64(
            self.d['project']['physical']['starting-field-x']['generalized-gauss-variables']['@wave-length'])
        
        settings.pulse_length = numpy.float64(
            self.d['project']['physical']['starting-field-x']['generalized-gauss-variables']['@pulse-length'])
        settings.phase = numpy.float64(
            self.d['project']['physical']['starting-field-x']['generalized-gauss-variables']['@phase'])
        settings.chirp = numpy.float64(
            self.d['project']['physical']['starting-field-x']['generalized-gauss-variables']['@chirp'])
        settings.x_curvature = numpy.float64(
            self.d['project']['physical']['starting-field-x']['generalized-gauss-variables']['@x-curvature'])
        settings.intensity = numpy.float64(
            self.d['project']['physical']['starting-field-x']['generalized-gauss-variables']['@intensity'])
        settings.x_width = numpy.float64(
            self.d['project']['physical']['starting-field-x']['generalized-gauss-variables']['@x-width'])
        settings.m = numpy.float64(
            self.d['project']['physical']['starting-field-x']['generalized-gauss-variables']['@m'])
        settings.n = numpy.float64(
            self.d['project']['physical']['starting-field-x']['generalized-gauss-variables']['@n'])

        if u'even-orders' in self.d['project']['physical']['dispersion-index']:
            settings.evenOrders = True
            settings.N0 = numpy.float64(
                self.d['project']['physical']['dispersion-index']['even-orders']['@N0'])
            settings.a = numpy.float64(
                self.d['project']['physical']['dispersion-index']['even-orders']['@a'])
            settings.b = numpy.float64(
                self.d['project']['physical']['dispersion-index']['even-orders']['@b'])

        settings.wv = numpy.float64(
            self.d['project']['physical']['nonlinearity']['@wv'])
        settings.tv = numpy.float64(
            self.d['project']['physical']['nonlinearity']['@Tv'])
        settings.n2e = numpy.float64(
            self.d['project']['physical']['nonlinearity']['@n2e'])
        settings.n2ev = numpy.float64(
            self.d['project']['physical']['nonlinearity']['@n2ev'])
        settings.hiX11 = numpy.float64(
            self.d['project']['physical']['nonlinearity']['@hiX11'])
        settings.hiX12 = numpy.float64(
            self.d['project']['physical']['nonlinearity']['@hiX12'])
        settings.use_cubic = str2bool(
            self.d['project']['physical']['nonlinearity']['@use-cubic'])
        settings.use_raman = str2bool(
            self.d['project']['physical']['nonlinearity']['@use-raman'])
        settings.use_square = str2bool(
            self.d['project']['physical']['nonlinearity']['@use-square'])
        settings.use_ion = str2bool(
            self.d['project']['physical']['nonlinearity']['@use-ion'])
        settings.use_cubic_envelope = str2bool(
            self.d['project']['physical']['nonlinearity']['@use-cubic-envelope'])
        settings.use_fft_for_nonlinear = str2bool(
            self.d['project']['physical']['nonlinearity']['@use-fft-for-nonlinear'])

        # computation variables
        settings.time_size = numpy.int32(
            self.d['project']['computation']['variables']['size']['@time'])
        settings.space_size = numpy.int32(
            self.d['project']['computation']['variables']['size']['@space'])
        settings.time_limit = numpy.float64(
            self.d['project']['computation']['variables']['limits']['@time'])
        settings.space_limit = numpy.float64(
            self.d['project']['computation']['variables']['limits']['@space'])
        settings.z = numpy.float64(
            self.d['project']['computation']['variables']['limit-z'])
        if u'Uniform' in self.d['project']['computation']['save-strategy']['@strategy']:
            settings.uniformSave = True
            settings.dz_save = numpy.float64(
                self.d['project']['computation']['save-strategy']['uniform-variables']['@dz'])
        if u'Uniform' in self.d['project']['computation']['z-strategy']['@strategy']:
            settings.z_strategy = "Uniform"
            settings.dz = numpy.float64(
                self.d['project']['computation']['z-strategy']['uniform-variables']['@dz'])
        elif u'Adaptive' in self.d['project']['computation']['z-strategy']['@strategy']:
            settings.z_strategy = "Adaptive"
            settings.dz = numpy.float64(
                self.d['project']['computation']['z-strategy']['adaptive-variables']['@dz'])
            settings.layer_number = numpy.int32(
                self.d['project']['computation']['z-strategy']['adaptive-variables']['@layer-number'])
            settings.possible_iteration_number = numpy.int32(
                self.d['project']['computation']['z-strategy']['adaptive-variables']['@possible-iteration-number'])
            settings.possible_error = numpy.float64(
                self.d['project']['computation']['z-strategy']['adaptive-variables']['@possible-error'])
        if u'Exponent' in self.d['project']['computation']['space-grid']['@distribution']:
            settings.exponentSpace = True
            settings.x0 = numpy.float64(
                self.d['project']['computation']['space-grid']['exponent-variables']['@x0'])
        if u'Uniform' in self.d['project']['computation']['time-grid']['@distribution']:
            settings.uniformGrid = True
            settings.pulse_shift = numpy.float64(
                self.d['project']['computation']['time-grid']['uniform-variables']['@pulse-shift'])


def main():
    pass

if __name__ == '__main__':
    main()