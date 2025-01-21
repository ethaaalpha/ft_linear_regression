import matplotlib.pyplot as pp
import csv, os, sys

data_file = "train.csv"

def runner():
    retu

def main():
    args = sys.argv

    if (len(args) != 2):
        print("python predict.py [value]", file=sys.stderr)
        return 2
    else:
        if (os.path.exists(data_file) and os.access(data_file, os.R_OK)):
            runner(args[1])
        else:
            print("The train.csv file do not exist or is not readable!")
            return 2

if __name__ == "__main__":
    main()
