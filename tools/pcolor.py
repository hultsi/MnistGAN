import sys, os.path, re
import numpy as np
import matplotlib.pyplot as plt

class Options:
    color = "b"
    linewidth = 2
    markersize = 12

txt_line = "----------------------------"

def doesExist(path):
    print(txt_line)
    print("Reading " + path)
    if (not os.path.isfile(path)):
        print("File doesn't seem to exist.")
        print(txt_line)
        return 0
    print(txt_line)
    return 1

def pcolor(fname, axs = None):
    splitted = re.split("\\\\|//|\\|/", fname)
    firstline = ""
    with open(fname) as f:
        firstline = f.readline().rstrip()
    columns = len(re.split("\t|\s+", firstline))
    data = np.transpose(np.loadtxt(fname, usecols=range(columns)))
    max = data.max()
    min = data.min()
    scaledDiff = (max - min) * .95
    axs.set_title(splitted[len(splitted) - 1])
    if (columns == 1):
        data = [data, data]
    return axs.pcolor(data, cmap = "jet", vmin = max-scaledDiff, vmax = min+scaledDiff)

def main(args):
    N = len(args)
    for i in range(N):
        if (not doesExist(args[i])):
            return 0
    fig, axs = plt.subplots(N, figsize=(10,6))
    if (N == 1):
        axs = [ axs ]

    fig.suptitle("Plots")

    for i in range(N):
        p = pcolor(args[i], axs[i])
        fig.colorbar(p, ax=axs[i])
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])