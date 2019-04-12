import numpy as np 

# read raw data from csv file
data = np.genfromtxt('IDEALtrainingDATA10_04.csv', delimiter=',')

# may need editing per file
# subset of each row to take as input
# first index is always 1 , because we need to skip header.
datapoints = data[1:,2:]

print("Original shape",datapoints.shape)

# Defines the number of datarows in each time sequence.
set_size = 7

# defines the number of overlapping datapoints ...
# ... between 2 consecutive rows
overlap_size = 5


# init variables
timeseries = []
tset = []
c = 0

# l should be a perfectly divisible by set_size
# l is the number of rows taken as input
l = datapoints.shape[0]


'''
Formula for calculating total number of iterations in the for loop:

(number of rows) + (number of times we read overlapping rows)

= 		l 		 +  int(l/set_size)*overlap_size

'''

for x in range(l + int(l/set_size)*overlap_size ):

	# if set size id reached,
	# add it to the timeseries
	if x % set_size == 0 and x != 0:

		if timeseries == []:
			timeseries = np.array([tset])
			
		else:			
			timeseries = np.append(timeseries,[tset],axis=0)
			
		tset = []
		c += 1

	# append each row to form the timesequence
	tset = np.concatenate((tset,datapoints[x - (c*overlap_size),:]))

	
print("Timeseries shape:",timeseries.shape)

# save the timeseries data as a csv file, (without header)
np.savetxt("tseries_ideal_7_5.csv",timeseries,delimiter=',', fmt='%f',)


