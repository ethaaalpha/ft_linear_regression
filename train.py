import os, sys
from tools import Analyzer, CSVManager

class Computer:
    learningRate = 1
    stop = 1e-8

    def __init__(self, data: list[tuple[int, int]]):
        self._data = data

    def compute(self) -> tuple[float, float]:
        """Return theta0 and theta1"""
        theta0: float = 0.0
        theta1: float = 0.0
        old_theta0: float = 1.0
        old_theta1: float = 1.0
        m: int = len(self._data)
        counter: int = 0

        while abs(old_theta1 - theta1) > self.stop and abs(old_theta0 - theta0) > self.stop:
            old_theta0, old_theta1 = theta0, theta1
            theta0 = self.__calculate_theta0(theta0, theta1, m)
            theta1 = self.__calculate_theta1(theta0, theta1, m)
            counter+=1
        print(f"Iterations: {counter}")
        return (theta0, theta1)

    def __calculate_theta0(self, theta0: float, theta1: float, m: int):
        gradient = (1/m) * sum((theta1*x+theta0-y) for x, y in self._data)
        return theta0 - self.learningRate * gradient

    def __calculate_theta1(self, theta0: float, theta1: float, m: int):
        gradient = (1/m) * sum(x * (theta1*x+theta0-y) for x, y in self._data)
        return theta1 - self.learningRate * gradient

def runner(filepath: str) -> list[tuple[float, float]]:
    csv = CSVManager()
    data = csv.load(filepath)
    y_values = [p[1] for p in data]

    if (len(data) == 0):
        print("The csv is empty!")
        return

    tool = Analyzer()
    brain = Computer(tool.normalize_list(data))

    thetas = brain.compute()
    min_y, max_y = min(y_values), max(y_values)

    csv.export("train.csv", thetas, (min_y, max_y))

    # x_values_norm = [p[0] for p in normalized_data]
    # y_values_pred = [computation[1] * x + computation[0] for x in x_values_norm]
    # denormalized = tool.denormalize(data, list(zip(x_values_norm, y_values_pred)))

    # pp.plot(x_values, y_values, 'o')
    # pp.plot(x_values_predict, y_values_predict, color='green')

    # print(f"algorithm accuracy: {brain.accuracy(y_values, y_values_denor):.2f}%")

    # pp.show()

def main():
    args = sys.argv

    if (len(args) != 2):
        print("python train.py [file.csv]", file=sys.stderr)
        return 2
    else:
        if (os.path.exists(args[1]) and os.access(args[1], os.R_OK)):
            runner(args[1])
        else:
            print("The file specified do not exist or is not readable!")
            return 2

if __name__ == "__main__":
    main()
