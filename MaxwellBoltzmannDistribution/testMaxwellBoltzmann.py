import numpy as np
import matplotlib.pyplot as plt

import MaxwellBoltzmann as mb

def MaxwellBoltzmann_2d_distribution( speed, particleMass, k_BT ):
    '''
    returns probability of speed according to two-dimensional
    Maxwell-Boltzmann distribution
    '''
    preFact = particleMass * speed / k_BT
    exponent = - particleMass * speed**2 / ( 2.0 * k_BT )
    p = preFact * np.exp( exponent )
    return p
    

k_BT = 1.
particleMass = 10.0

N = 10**4

#use MaxwellBoltzmann_2d_Naive to sample list of speeds and plot the histogram
speedList_Naive = np.array([ mb.MaxwellBoltzmann_2d_Naive( k_BT, particleMass) for  i in range(N) ])
plt.hist(speedList_Naive, bins = 50, density = 'true', label = 'MaxwellBoltzmann_2d_Naive')

#use MaxwellBoltzmann_2d_SemiPro to sample list of speeds and plot the histogram
speedList_SemiPro = np.array([ mb.MaxwellBoltzmann_2d_SemiPro( k_BT, particleMass) for  i in range(N) ])
plt.hist(speedList_SemiPro, bins = 50, density = 'true', histtype = 'step', linewidth = 2, label = 'MaxwellBoltzmann_2d_SemiPro')

#use MaxwellBoltzmann_2d_Pro to sample list of speeds and plot the histogram
speedList_Pro = np.array([ mb.MaxwellBoltzmann_2d_Pro( k_BT, particleMass) for  i in range(N) ])
plt.hist(speedList_Pro, bins = 50, density = 'true', histtype = 'step', linewidth = 2, label = 'MaxwellBoltzmann_2d_Pro')

#compare the distribution to the Maxwell-Boltzmann distribution
x_List = np.linspace( 0, max(speedList_Naive), 100 )
y_List = MaxwellBoltzmann_2d_distribution( x_List, particleMass, k_BT)
plt.plot(x_List, y_List, '-', color = 'purple', linewidth = 2, label = '$P(|\\vec{v}|) = \\frac{m |\\vec{v}|}{k_B T}\\exp\\left(-\\frac{m \\vec{v}^2}{2 k_B T}\\right)$')

plt.legend(loc = 'upper right')
plt.xlabel('$|\\vec{v}| = \\sqrt{v_x^2 + v_y^2}$')
plt.ylabel('$P(|\\vec{v}|)$')
plt.title('$k_B T = %.4f$, $m = %.4f$' % (k_BT, particleMass))
plt.show()
