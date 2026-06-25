import numpy as np
from scipy.signal import find_peaks

from tools.split_dataset import split_data
# RSS AND STD ALGORITHMS TO TRACK CENTROID MOTION FROM DELTA AND SUM BPMS

def find_peak_and_dip(sigma, period):
    '''
    BPMS SIGNAL HAVE DOUBLE THE INFO, BECAUSE SIGNAL TRAVELS BOTH DIRECTIONS, SO MUST FIND THE LIMITS OF THE PEAK AND GAP AS WELL

    '''
    std_noise = np.std(sigma)
    l = len(sigma)
    sigma_split = split_data(sigma, period=period)
    period = int(period)
    a_array = np.zeros(int(l/period))
    b_array = np.zeros(int(l/period))
    for i in range(int(l/period)):

        peaks, properties = find_peaks(sigma_split[i,:], height=3.5*std_noise, width =5,)

        if len(peaks) == 1:
            a_array[i] = properties["left_ips"][0] #+ peaks[0]
            b_array[i] = properties["right_ips"][0] #+ peaks[0]

        elif len(peaks) == 0:
            a_array[i] = np.nan
            b_array[i] = np.nan

        else:
            # If more than one peak is found, pick the tallest one
            j = np.argmax(properties["peak_heights"])
            a_array[i] = properties["left_ips"][j] #+ peaks[j]
            b_array[i] = properties["right_ips"][j] #+ peaks[j]
            
    a = np.nanmean(a_array)
    b = np.nanmean(b_array)     

    return a, b

 

    



def rss_position_from_delta_sigma(delta, sigma, sX, sigma_v, loops, period, type='up'):
    """
    Compute RSS position and uncertainty from Delta and Sigma arrays.

    Inputs:
      delta : 1D numpy array of shape (N,)   -- Δ = S_R - S_L
      sigma : 1D numpy array of shape (N,)   -- Σ = S_R + S_L
      sX    : float                          -- horizontal sensitivity (fraction/% per mm).
                                             # NOTE: supply consistent unit; if sX given as percent/mm (e.g. 2.0 %/mm),
                                             # convert to fraction/mm: sX = 0.02 (i.e. 2% -> 0.02)
      sigma_v : float                        -- RMS noise per ADC sample (same units as delta/sigma)

    Returns:
      x_rss : float          -- RSS position estimator (same length-scale as 1/sX input, e.g. mm)
      sigma_x : float        -- estimated standard uncertainty of x_rss
      details : dict         -- intermediate numbers: RSS_L, RSS_R, N
              """
    if type not in ['up', 'down']:
        raise ValueError("Invalid type. Must be 'up' or 'down'.")
    if type == 'up':
        a, b = find_peak_and_dip(sigma, period)
        a = int(np.round(a))
        b = int(np.round(b))
    if type == 'down':
        a, b = find_peak_and_dip(-sigma, period)
        a = int(np.round(a))
        b = int(np.round(b))
    # Reconstruct S_R and S_L from delta and sigma:
    S_R = 0.5 * (sigma + delta)
    S_L = 0.5 * (sigma - delta)

    #Split the whole data in loops
    S_R_split = np.empty((loops,period))
    for i in range(loops):  
        S_R_split[i,:] = S_R[i*period:(i+1)*period]

    S_L_split = np.empty((loops,period))
    for i in range(loops):  
        S_L_split[i,:] = S_L[i*period:(i+1)*period]

    x_evolution= np.empty(loops)
    sigma_x_evolution = np.empty(loops)
    # Compute RSS of each electrode:
    for i in range(loops):
        
      RSS_R = np.sqrt(np.sum(S_R_split[i,a:b].astype(np.float64) ** 2))
      RSS_L = np.sqrt(np.sum(S_L_split[i,a:b].astype(np.float64) ** 2))

      # position (Eq. (7) in Reiter & Singh)
      num = RSS_R - RSS_L
      den = RSS_R + RSS_L
      if den == 0:
          raise ValueError("RSS_R + RSS_L == 0 (division by zero). Check inputs.")
      x_evolution[i] = (1.0 / sX) * (num / den)

      # uncertainty estimate (propagation assuming sigma_RSS = sigma_v for each RSS)
      RSS_mean = den / 2.0
      # sigma_x ≈ sigma_v / ( sX * sqrt(2) * RSS_mean )
      sigma_x_evolution[i] = sigma_v / (sX * np.sqrt(2.0) * RSS_mean)


    return x_evolution, sigma_x_evolution


def std_position_from_delta_sigma(delta, sigma, sX, sigma_v, loops, period, type ='up'):
    """
    Compute RSS position and uncertainty from Delta and Sigma arrays.

    Inputs:
      delta : 1D numpy array of shape (N,)   -- Δ = S_R - S_L
      sigma : 1D numpy array of shape (N,)   -- Σ = S_R + S_L
      sX    : float                          -- horizontal sensitivity (fraction/% per mm).
                                             # NOTE: supply consistent unit; if sX given as percent/mm (e.g. 2.0 %/mm),
                                             # convert to fraction/mm: sX = 0.02 (i.e. 2% -> 0.02)
      sigma_v : float                        -- RMS noise per ADC sample (same units as delta/sigma)

    Returns:
      x_rss : float          -- RSS position estimator (same length-scale as 1/sX input, e.g. mm)
      sigma_x : float        -- estimated standard uncertainty of x_rss
    """
    if type not in ['up', 'down']:
        raise ValueError("Invalid type. Must be 'up' or 'down'.")
    if type == 'up':
        a, b = find_peak_and_dip(sigma, period)
        a = int(np.round(a))
        b = int(np.round(b))
    if type == 'down':
        a, b = find_peak_and_dip(-sigma, period)
        a = int(np.round(a))
        b = int(np.round(b))
    # Reconstruct S_R and S_L from delta and sigma:
    S_R = 0.5 * (sigma + delta)
    S_L = 0.5 * (sigma - delta)

    #Split the whole data in loops
    S_R_split = np.empty((loops,period))
    for i in range(loops):  
        S_R_split[i,:] = S_R[i*period:(i+1)*period]

    S_L_split = np.empty((loops,period))
    for i in range(loops):  
        S_L_split[i,:] = S_L[i*period:(i+1)*period]

    x_evolution= np.empty(loops)
    sigma_x_evolution = np.empty(loops)
    # Compute RSS of each electrode:
    for i in range(loops):
        
      STD_R = np.std(S_R_split[i,a:b].astype(np.float64))
      STD_L = np.std(S_L_split[i,a:b].astype(np.float64))

      # position (Eq. (7) in Reiter & Singh)
      num = STD_R - STD_L
      den = STD_R + STD_L
      if den == 0:
          raise ValueError("STD_R + STD_L == 0 (division by zero). Check inputs.")
      x_evolution[i] = (1.0 / sX) * (num / den)

      # uncertainty estimate (propagation assuming sigma_STD = sigma_v for each STD)
      STD_mean = den / 2.0
      # sigma_x ≈ sigma_v / ( sX * sqrt(2) * STD_mean )
      sigma_x_evolution[i] = sigma_v / (sX * np.sqrt(2.0) * STD_mean)

    
    return x_evolution, sigma_x_evolution
