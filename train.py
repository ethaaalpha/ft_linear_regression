import matplotlib.pyplot as pp
import csv, os, sys, time

class ToolBox:
    def mean_square(self, x, y) -> float:
        return (x-y)**2


class Brain:
    learningRate = 0.01

    def __init__(self, data: list[tuple[int, int]]):
        self._data = data

    def compute(self) -> tuple[float, float]:
        """Return theta0 and theta1"""
        theta0: float = 0 # also b
        theta1: float = 0 # also a
        m: int = len(self._data)

        for i in range(38):
            print(f"t0 {theta0} t1 {theta1}")
            theta0 = self.__calculate_theta0(theta1, theta0, m)
            theta1 = self.__calculate_theta1(theta1, theta0, m)

        return theta0, theta1

    def __calculate_theta0(self, a: float, b: float, m: int):
        gradient = 0

        for values in self._data:
            x = values[0]
            y = values[1]
            gradient += (a*x+b-y)
        gradient /= m

        print(round(b - self.learningRate * gradient, 5))
        return round(b - self.learningRate * gradient, 5)

    def __calculate_theta1(self, a: float, b: float, m: int):
        gradient = 0

        for values in self._data:
            x = values[0]
            y = values[1]
            gradient += x*(a*x+b-y)
            # print(f"grad: {gradient}")
        gradient /= m
        # print(f"grad_div: {gradient}")

        return round(a - self.learningRate * gradient, 5)


def load(filepath: str) -> list[tuple[int, int]]:
    """Load the CSV file where tuple[0] = x and tuple[1] = y"""
    result: list[tuple[int, int]] = list()

    with open(filepath, 'r') as file:
        data = list(csv.reader(file, delimiter=","))

        if (len(data) > 1):
            for row in data[1:]:
                if (len(row) == 2 and row[0].isdigit() and row[1].isdigit()):
                    # print(f"{int(row[0])} | {int(row[1])}")
                    result.append((int(row[0]), int(row[1])))
    return result

def runner(filepath: str):
    data = load(filepath)
    brain = Brain(data)

    if (len(data) == 0):
        print("The csv is empty!")
        return

    res = brain.compute()
    # print(res)
    x_values = [point[0] for point in data]
    y_values = [point[1] for point in data]
    y_values_pred = [res[0] + res[1] * x for x in x_values]

    # pp.scatter(x_values, y_values)
    # pp.plot(x_values, y_values_pred)
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
