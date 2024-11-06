from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NumpyroResultSet:
    """
    Result set for numpyro models.

    Args:
        samples: MCMC samples.
        grouped_by_chain: Whether samples are grouped by chain.
        num_chains: Number of chains.
        num_samples: Number of samples.
    """

    samples: Dict[str, np.ndarray]

    grouped_by_chain: bool
    num_chains: int
    num_samples: int

    def get_samples(self, *, group_by_chain: bool = False) -> Dict[str, np.ndarray]:
        if group_by_chain and self.grouped_by_chain:
            return self.samples

        if group_by_chain and not self.grouped_by_chain:
            return {
                k: np.copy(v).reshape((self.num_chains, self.num_samples) + v.shape[1:])
                for k, v in self.samples.items()
            }

        if not group_by_chain and self.grouped_by_chain:
            return {
                k: np.copy(v).reshape((self.num_chains * self.num_samples,) + v.shape[2:])
                for k, v in self.samples.items()
            }

        return {k: np.copy(v) for k, v in self.samples.items()}

    def summary(self, **kwargs) -> pd.DataFrame:
        """
        Prints summary.

        Args:
            **kwargs: Kwargs passed to :func:`arviz.summary`.

        Returns:
            Nothing.
        """

        import arviz

        return arviz.summary(self.samples, **kwargs)

    def __getstate__(self):
        return {
            "samples": {k: np.array(v) for k, v in self.get_samples(group_by_chain=self.grouped_by_chain).items()},
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "grouped_by_chain": self.grouped_by_chain,
        }

    def __setstate__(self, state):
        samples = {k: np.array(v) for k, v in state["samples"].items()}
        object.__setattr__(self, "samples", samples)

        for attr in ["num_samples", "num_chains", "grouped_by_chain"]:
            object.__setattr__(self, attr, state[attr])

        return
