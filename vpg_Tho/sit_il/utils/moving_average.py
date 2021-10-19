import numpy as np


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Return the moving average of the elements in a 1-D array.

    Args:
        data (numpy.ndarray): Input 1-D array.
        window_size (int): The size of sliding window.

    Returns:
        numpy.ndarray: Moving average of the elements in `data`.

    Examples:
        >>> data = np.array([10,5,8,9,15,22,26,11,15,16,18,7])
        >>> moving_average(data, 4)
        array([ 8.  ,  9.25, 13.5 , 18.  , 18.5 , 18.5 , 17.  , 15.  , 14.  ])
    """
    return np.convolve(data, np.ones(window_size), "valid") / window_size
