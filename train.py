import matplotlib.pyplot as pp
import csv, os, sys

class Brain:
    def __init__(self, data: list[tuple[int, int]]):
        self._data = data
    
    def _calculate_theta0():
        return 0
    
    def _calculate_theta1():
        return 0

def load(filepath: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = list()

    with open(filepath, 'r') as file:
        data = list(csv.reader(file, delimiter=","))

        if (len(data) > 1):
            for row in data[1:]:
                if (len(row) == 2 and row[0].isdigit() and row[1].isdigit()):
                    result.append((int(row[0]), int(row[1])))
    return result

def runner(filepath: str):
    data = load(filepath)

    if (len(data) == 0):
        print("The csv is empty!")
        return
    for d in data:
        pp.plot(d[0], d[1], 'o')
        print(d)
    pp.show()
    

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
