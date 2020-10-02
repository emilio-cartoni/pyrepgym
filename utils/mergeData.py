import numpy as np
import glob

files = glob.glob("./data/*")

allData = None
for f in files:
    dataFile = np.loadtxt(f)
    print("Loading... {}-{}".format(f, dataFile.shape))
    if allData is None:
        allData = dataFile
    else:
        allData = np.vstack([allData, dataFile])

print(allData.shape)
np.savetxt('mergedData', allData)



