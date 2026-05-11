import numpy as np



def interp_cross(x0, x1, y0, y1, thr):
    """Return interpolated x where line (x0,y0)-(x1,y1) crosses thr.
       Assumes y0 != y1 and thr between y0 and y1."""
    return x0 + (thr - y0) / (y1 - y0)

def fwhm_for_row(bunch, thr, direction='above'):
    """
    Find first rising crossing and last falling crossing around the peak/gap:
    - direction='above' looks for first index where value >= thr then next where <= thr.
    - direction='below' looks for first index where value <= thr then next where >= thr.
    Returns width (float) in sample units (interpolated) or None if not found.
    """
    # ensure numpy array
    s = np.asarray(bunch)
    n = s.size

    if direction == 'above':
        # boolean crossing array: True where s >= thr
        mask = s >= thr
    else:
        mask = s <= thr

    true_idxs = np.flatnonzero(mask)
    if true_idxs.size == 0:
        return None  # never reaches threshold

    # first contiguous block that contains the first crossing
    start = true_idxs[0]

    # find end: the first index after start where mask becomes False
    # look from start to end
    after = np.flatnonzero(~mask[start+1:])  # indices relative to start+1
    if after.size == 0:
        # never falls back below/above threshold after start
        end = n - 1
        end_rel = end
    else:
        end = start + 1 + after[0]
        end_rel = end

    # interpolate left crossing: between start-1 and start (if start > 0)
    if start == 0:
        left_x = 0.0
    else:
        y0, y1 = s[start-1], s[start]
        # if y1 == y0 (flat), take start as crossing
        if y1 == y0:
            left_x = start
        else:
            left_x = interp_cross(start-1, start, y0, y1, thr)

    # interpolate right crossing: between end-1 and end (if end > 0)
    if end == 0:
        right_x = 0.0
    else:
        # If end is last index and mask[end] is still True, try to interpolate using end-1->end.
        y0, y1 = s[end-1], s[end]
        if y1 == y0:
            right_x = end
        else:
            # we want the point where it drops back across thr; note the segment chosen should bracket thr
            right_x = interp_cross(end-1, end, y0, y1, thr)

    width = right_x - left_x
    return width

def compute_fwhms(up_split_bunches):
    """
    up_split_bunches: 2D array shape (nrows, ncols)
    Returns (gap_FWHMs, peak_FWHMs) as numpy arrays of floats (NaN where not computable).
    """
    up_split_bunches = np.asarray(up_split_bunches)
    nrows = up_split_bunches.shape[0]

    gap_FWHMs = np.full(nrows, np.nan)
    peak_FWHMs = np.full(nrows, np.nan)

    for i, bunch in enumerate(up_split_bunches):
        # center the bunch to mean 0
        s = bunch - np.mean(bunch)

        # thresholds like your original code:
        half_min = np.min(s) / 2.0
        half_max = np.max(s) / 2.0

        # gap: direction 'below' (look for s <= half_min then return to >= half_min)
        gap_w = fwhm_for_row(s, half_min, direction='below')
        if gap_w is None:
            gap_FWHMs[i] = 0.0
        else:
            # keep the same rule you had: ignore spurious very wide ones (>30 samples)
            gap_FWHMs[i] = gap_w if gap_w < 100 else 0.0

        # peak: direction 'above'
        peak_w = fwhm_for_row(s, half_max, direction='above')
        if peak_w is None:
            peak_FWHMs[i] = 0.0
        else:
            peak_FWHMs[i] = peak_w

    return gap_FWHMs, peak_FWHMs


def baseline(data, stop):
    cycles = np.shape(data)[0]
    period = np.shape(data)[1]

    if stop is None:
        stop = 0.05*cycles
    
    maxs_indices = []
    mins_indices = []
    for i in range(stop):
        bunch = data[i,:]
        maxs_indices.append([np.argmax(bunch),np.max(bunch)])
        mins_indices.append([np.argmin(bunch),np.max(bunch)])  

    maxs_indices = np.array(maxs_indices)
    mins_indices = np.array(mins_indices)
    type(maxs_indices)
    np.shape(maxs_indices)
    peak_avg_idx = int(np.round(np.mean(maxs_indices[:,0])))
    dip_avg_idx  = int(np.round(np.mean(mins_indices[:,0])))
    

    FWHM_peaks, FWHM_dips = compute_fwhms(data[:stop,:])
    FWHM_peak_avg = int(np.round(np.mean(FWHM_peaks))) #INSTEAD OF THIS METHOD, IT COULD BE USEFUL TO BIN THE DATA AND USE THE MODAL VALUE
    FWHM_dip_avg = int(np.round(np.mean(FWHM_dips)))
    sigma_peak = FWHM_peak_avg / (2 * np.sqrt(2 * np.log(2)))
    sigma_dip = FWHM_dip_avg / (2 * np.sqrt(2 * np.log(2)))

    peak_amplitude = np.max(data[:stop,:]) - np.mean(data[:stop,:])
    dip_amplitude  = np.mean(data[:stop,:]) - np.min(data[:stop,:])

    baseline_signal = np.ones(period) * np.mean(data[:stop,:])

    baseline_signal[peak_avg_idx-3*int(sigma_peak):peak_avg_idx+3*int(sigma_peak)] += peak_amplitude * np.exp(-0.5 * ((np.arange(peak_avg_idx-3*int(sigma_peak), peak_avg_idx+3*int(sigma_peak)) - peak_avg_idx) / sigma_peak) ** 2)
    baseline_signal[dip_avg_idx-3*int(sigma_dip):dip_avg_idx+3*int(sigma_dip)] -= dip_amplitude * np.exp(-0.5 * ((np.arange(dip_avg_idx-3*int(sigma_dip), dip_avg_idx+3*int(sigma_dip)) - dip_avg_idx) / sigma_dip) ** 2)

    return baseline_signal
    

