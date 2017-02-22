# calculate the area and circumference of a circle from its radius

import math
import pylab

IniNum = int(raw_input("Your trial number:"))
aPlot = []
#aPlotIndex = 0
while IniNum != 1:
    aPlot.append(IniNum)
    if IniNum%2 == 0:
        IniNum /= 2
    else:
        IniNum = IniNum * 3 + 1
else:
    aPlot.append(IniNum)

print aPlot
print len(aPlot)
pylab.plot(aPlot,'r')
pylab.show()

