import numpy as np
x=np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]],np.float64)
z=np.array([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]],np.float64)
a=np.array([736,-32+77.2548j,-32-32j,-32+13.2548j,-32,-32-13.2548j,-32-32j,-32-77.2548j],dtype=complex)



z=np.array([0,1,2,3,4,3,2,1],dtype=complex)
y=np.fft.ifft(z)
y=y*8
print y
raw_input()
