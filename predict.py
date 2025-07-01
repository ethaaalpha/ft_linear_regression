import os, sys
from tools import Analyzer, CSVManager

data_file = "train.csv"

def runner(value: int, thetas: tuple[float, float], x_min_max, y_min_max: tuple[float, float]):
    tool = Analyzer()
    normalized_value = tool.normalize(x_min_max[0], x_min_max[1], value)
    normalized_result = tool.linear_function(thetas[1], normalized_value, thetas[0])
    denormalized_result = tool.denormalize(y_min_max[0], y_min_max[1], normalized_result)

    print(f"Predicted price for {value}km: {denormalized_result:.2f}")
    return

def parser(value: str):
    csv = CSVManager()

    if not value.isnumeric() or int(value) < 0:
        print("Invalid value passed must be a positive number!")
        return

    values = csv.load("train.csv")
    if (len(values) != 3 or len(values[0]) != 2 or len(values[1]) != 2):
        print("Invalid train.csv, please run training first!")
        return
    
    thetas = values[0]
    x_min_max = values[1]
    y_min_max = values[2]

    runner(int(value), thetas, x_min_max, y_min_max)

def main():
    args = sys.argv

    if (len(args) != 1):
        print("use: python predict.py", file=sys.stderr)
        return 2
    else:
        if (os.path.exists(data_file) and os.access(data_file, os.R_OK)):
            print("Please enter a specific mileage in KM: ", end="")
            parser(input())
        else:
            print("The train.csv file do not exist or is not readable!")
            return 2

if __name__ == "__main__":
    main()
