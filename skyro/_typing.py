from typing import Literal, Union

from numpy import ndarray
from pandas import DataFrame, Index, Series

ArrayLike = Union[ndarray, Series, DataFrame, None]
IndexLike = Union[ndarray, Index, None]


OutputType = Literal["np.ndarray", "pd.Series", "pd.DataFrame"]
