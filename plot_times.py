

import numpy as np
import matplotlib.pyplot as plt

		###################################### Harris Corner Plot ##################################
		#                          This Plots the performance of the 5 version of the              #
		#                          parallelel Harris Corner detection algorithm                    #
		#                          												                   #
		###################################### Harris Corner Plot ##################################

OY = [0.101152591944, 0.11575482583, 0.0678665053844, 0.0595110816956, 0.0581274104118, 0.0575940756798]
OX = ['v1: No buffer','v2: Buffer w/ CPU transfer','v3: No CPU transfers, w/ events','v4: Constants memory, no evnts',
 'v5: Added loops', 'final: Banks and variable buffer']

serial_detection = 0.0889417750835


fig = plt.figure()

width = .35
ind = np.arange(len(OY))
plt.bar(ind, OY)
plt.xticks(ind + width / 2, OX, fontsize = 9)
plt.title('Performance Times for Each Version of Corner Detection: Serial time is 0.088 sec')
plt.ylabel('Time')
plt.axhline(y = serial_detection, xmin = 0, xmax = 100, linewidth = 4, color='r', alpha = .5
            , hold = None, label = 'Serial Performance')
plt.legend(loc = 'upper right')

fig.autofmt_xdate()


plt.savefig("corner_detection_performance.jpg")

##############################################################################################################


		################################ Point Matching Algorithm ##################################
		#                          This Plots the performance of the 5 version of the              #
		#                          parallelel point matching algorithm  		                   #
		#                          												                   #
		################################## Point Matching Algorithm ################################



OY2 = [0.664073355198, 0.66395629406, 0.5422777915, 0.515796060562]
OX2 = ['v1: 2 Pass Naive', 'v2: 2 Pass w/ Preprocess', 'v3: Online Naive', 'v4: Online w/ Preprocess']
serial_matching = 60.4599671364

fig = plt.figure()
width = .35
ind = np.arange(len(OY2))
plt.bar(ind, OY2)
plt.xticks(ind + width / 2, OX2, fontsize = 9)
plt.title('Performance Times for Each Version of Corner Matching: Serial time is 60.459 sec')
plt.ylabel('Time')

fig.autofmt_xdate()
plt.savefig("corner_matching_performance.jpg")
plt.show()
##############################################################################################################





