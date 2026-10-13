def history_times(now, history_nfe):
    delta = 1/history_nfe.float()
    reference = (now-delta).clamp_min(0)
    previous = (reference-delta).clamp_min(0)
    return previous, reference, now
