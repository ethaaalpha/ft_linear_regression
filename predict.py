import matplotlib.pyplot as pp
import os, sys
from tools import Analyzer, CSVManager

data_file = "train.csv"

def linear_function(a: float, x: float, b: float):
    return a * x + b

def runner(value: int, thetas: tuple[float, float], y_min_max: tuple[float, float]):
    tool = Analyzer()
    normalized_value = tool.normalize(y_min_max[0], y_min_max[1], value)
    normalized_result = linear_function(thetas[1], normalized_value, thetas[0])
    denormalized_result = tool.denormalize(y_min_max[0], y_min_max[1], normalized_result)

    print(normalized_result)
    print(denormalized_result)
    return

def parser(value: str):
    csv = CSVManager()

    if not value.isalnum() or int(value) < 0:
        print("Invalid value passed must be a positive number!")
        return

    values = csv.load("train.csv")
    if (len(values) != 2 or len(values[0]) != 2 or len(values[1]) != 2):
        print("Invalid train.csv, please run training first!")
        return
    
    thetas = values[0]
    y_min_max = values[1]

    runner(int(value), thetas, y_min_max)

def main():
    args = sys.argv

    if (len(args) != 2):
        print("python predict.py [value]", file=sys.stderr)
        return 2
    else:
        if (os.path.exists(data_file) and os.access(data_file, os.R_OK)):
            parser(args[1])
        else:
            print("The train.csv file do not exist or is not readable!")
            return 2

if __name__ == "__main__":
    main()
