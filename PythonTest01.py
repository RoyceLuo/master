import numpy as np
import matplotlib.pyplot as plt

print("Hello")
print("A change has been made!")

x = np.linspace(0,10,100)
y = 1/(5**2-x**2-1j*2*x)

plt.figure()
plt.plot(x,np.imag(y))
plt.show()

