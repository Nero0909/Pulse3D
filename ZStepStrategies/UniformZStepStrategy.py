from GlobalSettings import Settings


class UniformZStepStrategy:
    def __init__(self):
        self.uniform_dz = Settings.dz
        self.iterations = 0

    def calculate_dz(self, current_dz, iteration_number, errors):
        if current_dz < 0:
            current_dz = self.uniform_dz

        return current_dz

    def need_update_dz(self, iteration_number, errors):
        return False
