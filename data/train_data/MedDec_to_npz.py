"""
MedDec_to_npz.py  —  Medical Segmentation Decathlon (MSD)
==========================================================
Converts MSD CT + MRI tasks from the DPX server to a single .npz archive.

Tasks included
--------------
===  ==============================  ========  =====================
Tag  Task                            Modality  Channels
===  ==============================  ========  =====================
train_ct   Task03 Liver              CT        1
train_ct   Task06 Lung               CT        1
train_ct   Task07 Pancreas           CT        1
train_ct   Task09 Spleen             CT        1
train_mri  Task01 Brain Tumour       MRI       4 (FLAIR/T1w/T1gd/T2w)
train_mri  Task05 Prostate           MRI       2 (T2/ADC)
===  ==============================  ========  =====================

Multi-channel MRI: each channel becomes a separate entry (same label,
channel index appended to the patient ID).

Server-side usage
-----------------
::

    loader = MedDec_to_npz()
    loader.to_npz('data/test_data/MSD')

The single .npz (CT + MRI, modality-tagged) is loaded at training time
via DataLoader_npz, which passes the tag to universal_normalization().

Data loading
------------
Loading is handled by DPXSession in dpx_loader.py (server-specific).
See that module for instructions on replacing the loader with your own.
Note: data on the server is already cropped — no cropping is applied here.
"""

import numpy as np
import tensorflow as tf

from data.train_data.dpx_loader import DPXSession, resample_and_save


class MedDec_to_npz:
    """
    Offline converter for MSD CT and MRI tasks.

    Parameters
    ----------
    max_img : int
        Maximum number of patients to load *per modality group* (CT / MRI).
    """

    def __init__(self, max_img=1000):
        self.max_img = max_img
        self.dataset: dict = {}
        self._pull_data()

    # ------------------------------------------------------------------
    # Patient selection  (all server-side loading is in DPXSession)
    # ------------------------------------------------------------------

    def _pull_data(self):
        session = DPXSession()

        # CT tasks: Task03 Liver, Task06 Lung, Task07 Pancreas, Task09 Spleen
        session.get('STAG:train_ct', 'img.nii.gz', 'labels.nii.gz')
        self.dataset.update(
            session.load_volumes(max_img=self.max_img, modality='CT', id_prefix='ct')
        )

        # MRI tasks: Task01 Brain (4ch), Task05 Prostate (2ch)
        session.reset()
        session.get('STAG:train_mri', 'img.nii.gz', 'labels.nii.gz')
        self.dataset.update(
            session.load_volumes(max_img=self.max_img, modality='MRI', id_prefix='mri')
        )

    # ------------------------------------------------------------------
    # NPZ export
    # ------------------------------------------------------------------

    def to_npz(self, output_path: str):
        """Resample all volumes to 1 mm isotropic spacing and save as .npz."""
        resample_and_save(self.dataset, output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MSD → .npz converter")
    parser.add_argument("--output",  default="MSD")
    parser.add_argument("--max-img", type=int, default=1000)
    args = parser.parse_args()

    loader = MedDec_to_npz(max_img=args.max_img)
    loader.to_npz(args.output)