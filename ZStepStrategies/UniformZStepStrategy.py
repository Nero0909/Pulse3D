from GlobalSettings import Settings


class UniformZStepStrategy:
    def __init__(self):
        self.uniform_dz = Settings.dz
        self.iterations = 0

    def calculateDz(self, current_dz):
        if current_dz < 0:
            current_dz = self.uniform_dz

        return current_dz

    def needUpdateDz(self, iteration_number):
        return False
