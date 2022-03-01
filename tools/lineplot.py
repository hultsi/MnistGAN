#########################################################
# Takes in N number of command line arguments
# Each argument must be a valid path of a text file
# Each line in the text file must contain single value
# Plots line plots from the files
#########################################################

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

def linePlot(fname, axs = None):
    splitted = re.split("\\\\|//|\\|/", fname)
    lines = np.loadtxt(fname, delimiter="  ", unpack=True)
    size = len(lines.shape)
    if (axs == None):
        if (size == 1):
            return plt.plot(np.arange(len(lines)),lines, color = Options.color, linewidth = Options.linewidth, markersize = Options.markersize)
        elif (size == 2):
            return plt.plot(lines[0], lines[1], color = Options.color, linewidth = Options.linewidth, markersize = Options.markersize)
    else:
        axs.set_title(splitted[len(splitted) - 1])
        if (size == 1):
            return axs.plot(np.arange(len(lines)),lines, color = Options.color, linewidth = Options.linewidth, markersize = Options.markersize)
        elif (size == 2):
            return axs.plot(lines[0], lines[1], color = Options.color, linewidth = Options.linewidth, markersize = Options.markersize)
    return None

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
        linePlot(args[i], axs[i])
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])