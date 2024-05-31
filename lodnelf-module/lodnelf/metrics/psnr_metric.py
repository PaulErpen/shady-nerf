import math


def calculate_psnr(loss: float) -> float:
    if loss == 0:
        return float("inf")
    return -10.0 * math.log(loss) / math.log(10.0)
