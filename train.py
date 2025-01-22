import matplotlib.pyplot as pp
import csv, os, sys

class Normalizer:
    def normalize(self, data: list[tuple[int, int]]) -> list[tuple[int, int]]:
        min_x, max_x = min([x for x,_ in data]), max([x for x,_ in data])
        min_y, max_y = min([y for _,y in data]), max([y for _,y in data])
        all_x_normalized = [(x - min_x) / (max_x - min_x) for x, _ in data]
        all_y_normalized = [(y - min_y) / (max_y- min_y) for _, y in data]

        return list(zip(all_x_normalized, all_y_normalized))

    def denormalize(self, original_data: list[tuple[int, int]], normalized_data: list[tuple[int, int]]) -> list[tuple[int, int]]:
        min_x, max_x = min([x for x,_ in original_data]), max([x for x,_ in original_data])
        min_y, max_y = min([y for _,y in original_data]), max([y for _,y in original_data])
        all_x_reverted = [x * (max_x - min_x) + min_x for x,_ in normalized_data]
        all_y_reverted = [y * (max_y - min_y) + min_y for _,y in normalized_data]
        
        return list(zip(all_x_reverted, all_y_reverted))

class Computer:
    learningRate = 0.01
    stop = 1e-12

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
        print(f"gradient descent iterate : {counter} times")
        return theta0, theta1

    def __calculate_theta0(self, theta0: float, theta1: float, m: int):
        gradient = (1/m) * sum((theta1*x+theta0-y) for x, y in self._data)
        return theta0 - self.learningRate * gradient

    def __calculate_theta1(self, theta0: float, theta1: float, m: int):
        gradient = (1/m) * sum(x * (theta1*x+theta0-y) for x, y in self._data)
        return theta1 - self.learningRate * gradient

def runner(filepath: str):
    data = load(filepath)

    if (len(data) == 0):
        print("The csv is empty!")
        return

    tool = Normalizer()
    normalized_data = tool.normalize(data)
    brain = Computer(normalized_data)
    res = brain.compute()
    print(f"thetas : {res}")

    x_values = [p[0] for p in data]
    y_values = [p[1] for p in data]
    pp.plot(x_values, y_values, 'o')

    x_values_norm = [p[0] for p in normalized_data]
    y_values_pred = [res[1] * x + res[0] for x in x_values_norm]
    denormalized = tool.denormalize(data, list(zip(x_values_norm, y_values_pred)))

    x_values_denor = [p[0] for p in denormalized]
    y_values_denor = [p[1] for p in denormalized]
    pp.plot(x_values_denor, y_values_denor, color='green')
    pp.show()

def load(filepath: str) -> list[tuple[int, int]]:
    """Load the CSV file where tuple[0] = x and tuple[1] = y"""
    result: list[tuple[int, int]] = list()

    with open(filepath, 'r') as file:
        data = list(csv.reader(file, delimiter=","))

        if (len(data) > 1):
            for row in data[1:]:
                if (len(row) == 2 and row[0].isdigit() and row[1].isdigit()):
                    result.append((int(row[0]), int(row[1])))

    return result

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
