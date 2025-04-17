import numpy as np

# def sigmoid_ramp_up(steps, ramp_up_length):
#     if (steps >= ramp_up_length):
#         return 1.0
#     else:
#         return np.exp(-5*(1.0 - steps / ramp_up_length)**2)

def sigmoid_ramp_up(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def cosine_ramp_down(current, ramp_down_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    # assert 0 <= current <= rampdown_length
    return 0.5 * (np.cos(np.pi * current / ramp_down_length) + 1)