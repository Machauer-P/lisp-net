"""
TotalSeg_to_npz.py
==================
Converts TotalSegmentator CT data from the DPX server to .npz.

Server-side usage
-----------------
::

    loader = TotalSeg_to_npz(mode='train_combined')
    loader.to_npz('data/test_data/TotalSeg')

The .npz is then loaded at training time via DataLoader_npz.

Data loading
------------
Loading is handled by DPXSession in dpx_loader.py (server-specific).
See that module for instructions on replacing the loader with your own.
Note: data on the server is already cropped — no cropping is applied here.
"""

import numpy as np
import tensorflow as tf

from data.train_data.dpx_loader import DPXSession, resample_and_save


class TotalSeg_to_npz:
    """
    Offline converter for TotalSegmentator CT data.

    Parameters
    ----------
    mode : str
        DPX STAG suffix, e.g. 'train_combined' → loads 'STAG:train_combined'.
    max_img : int
        Maximum number of patients to load.
    """

    def __init__(self, mode='train_combined', max_img=1000):
        self.mode    = mode
        self.max_img = max_img
        self.dataset: dict = {}
        self._pull_data()

    # ------------------------------------------------------------------
    # Patient selection  (all server-side loading is in DPXSession)
    # ------------------------------------------------------------------

    def _pull_data(self):
        session = DPXSession()
        session.get('STAG:' + self.mode, 'ct.nii.gz', 'labels.nii.gz')
        self.dataset.update(
            session.load_volumes(max_img=self.max_img, modality='CT')
        )

    # ------------------------------------------------------------------
    # NPZ export
    # ------------------------------------------------------------------

    def to_npz(self, output_path: str):
        """Resample all volumes to 1 mm isotropic spacing and save as .npz."""
        resample_and_save(self.dataset, output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TotalSegmentator → .npz converter")
    parser.add_argument("--mode",    default="train_combined")
    parser.add_argument("--output",  default="TotalSeg")
    parser.add_argument("--max-img", type=int, default=1000)
    args = parser.parse_args()

    loader = TotalSeg_to_npz(mode=args.mode, max_img=args.max_img)
    loader.to_npz(args.output)