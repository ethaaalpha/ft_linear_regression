import os, sys
from matplotlib import pyplot as pl
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

def display(initial_data: list[tuple[float, float]], result: list[tuple[float,float]]):
    tool = Analyzer()
    denormalized_result = tool.denormalize_list(initial_data, result)
    x_values = [p[0] for p in initial_data]
    y_values = [p[1] for p in initial_data]
    y_predictions = [p[1] for p in denormalized_result]

    pl.figure("ft_linear_regression")
    pl.plot(x_values, y_values, 'o')
    pl.plot(x_values, y_predictions, color='green')
    pl.title(f"algorithm accuracy: {tool.accuracy(y_values, y_predictions):.2f}%")
    pl.xlabel("km")
    pl.ylabel("price")
    pl.show()

def runner(data: list[tuple[float, float]]):
    tool = Analyzer()

    normalized_data = tool.normalize_list(data)
    x_values = [p[0] for p in data]
    y_values = [p[1] for p in data]

    brain = Computer(normalized_data)

    thetas = brain.compute()
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    CSVManager().export("train.csv", thetas, (min_x, max_x), (min_y, max_y))
    y_predictions = [tool.linear_function(thetas[1], p[0], thetas[0]) for p in normalized_data]
    display(data, list(zip(y_values, y_predictions)))

def loader(filepath: str):
    data = CSVManager().load(filepath)

    if (len(data) == 0):
        print("The csv is empty!")
    else:
        runner(data)

def main():
    args = sys.argv

    if (len(args) != 2):
        print("python train.py [file.csv]", file=sys.stderr)
        return 2
    else:
        if (os.path.exists(args[1]) and os.access(args[1], os.R_OK)):
            loader(args[1])
        else:
            print("The file specified do not exist or is not readable!")
            return 2

if __name__ == "__main__":
    main()