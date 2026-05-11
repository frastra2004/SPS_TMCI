import numpy as np
def split_data(data, period):
    '''Returns the array split in cycles'''
    l = len (data)
    cycles = int(l/period)
    data_split = np.empty((cycles,period))

    for i in range(cycles):
        data_split[i,:] = data[i*1000:(i+1)*1000]

    return data_split