from typing import Any

import numpy as np
from numpy.typing import NDArray

def mel_filterbank() -> NDArray[np.float32]: ...
def log_mel(clip: NDArray[np.floating[Any]]) -> NDArray[np.float32]: ...
