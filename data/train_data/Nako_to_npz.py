"""
Nako_to_npz.py
==============
Converts NAKO (GPUnet) body / head MRI from the DPX server to .npz.

Server-side usage
-----------------
::

    loader = Nako_to_npz(mode='training_body')
    loader.to_npz('data/test_data/GPUnet_body')

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


class Nako_to_npz:
    """
    Offline converter for NAKO (GPUnet) MRI data.

    Parameters
    ----------
    mode : str
        'training_body'     — whole-body 3-D GRE Dixon MRI
        'training_combined' — body + head
        'training_head'     — T1 / T1km / T2 head MRI
        'testing_head'      — fixed test patients (head)
        'testing_body'      — fixed test patients (body)
    max_img : int
        Maximum number of patients per patient group to load.
    """

    def __init__(self, mode='training_body', max_img=1000):
        self.mode    = mode
        self.max_img = max_img
        self.dataset: dict = {}
        self._pull_data()

    # ------------------------------------------------------------------
    # Patient selection  (all server-side loading is in DPXSession)
    # ------------------------------------------------------------------

    def _pull_data(self):
        session = DPXSession()

        if self.mode == 'testing_head':
            for pid in ['16641588', '16300381']:
                session.get(pid, 'T1_crop.nii',   'freesurfer/wmparc_atlas.nii.gz')
                session.get(pid, 'T1km_crop.nii',  'freesurfer/wmparc_atlas.nii.gz')
                session.get(pid, 'T2_crop.nii',    'freesurfer/wmparc_atlas.nii.gz')

        elif self.mode == 'testing_body':
            for c, p in enumerate(['104177', '104178']):
                if c == self.max_img:
                    break
                session.get(p, '3D_GRE_TRA_4/3D_GRE_TRA_W_COMPOSED*',
                               'wholebody/wbcomp2.nii.gz')

        elif self.mode in ('training_combined', 'training_body'):
            for c, p in enumerate(['104171', '104172', '104173', '104174',
                                    '104175', '104176', '104179']):
                if c == self.max_img:
                    break
                session.get(p, '3D_GRE_TRA_4/3D_GRE_TRA_W_COMPOSED*',
                               'wholebody/wbcomp2.nii.gz')

            if self.mode == 'training_combined':
                for c, p in enumerate(['074038b7ec', '10283710', '10466628',
                                        '10499178',   '10951992', '11317731',
                                        '11307671',   '11373500', '11488188',
                                        '11490352',   '11555403']):
                    if c == self.max_img:
                        break
                    session.get(p, 'T1_crop.nii',  'freesurfer/wmparc_atlas.nii.gz')
                    session.get(p, 'T1km_crop.nii', 'freesurfer/wmparc_atlas.nii.gz')
                    session.get(p, 'T2_crop.nii',   'freesurfer/wmparc_atlas.nii.gz')

        elif self.mode == 'training_head':
            patients = [
                '074038b7ec', '10283710', '10466628', '10499178',
                '10951992',   '11317731', '11307671', '11373500',
                '11488188',   '11490352', '11555403', '11623891',
                '11749127',   '15042311', '15104554', '15505273',
                '15548827',   '15966696', '16529605', '16856738',
            ]
            for c, p in enumerate(patients):
                if c == self.max_img:
                    break
                session.get(p, 'T1_crop.nii',  'freesurfer/wmparc_atlas.nii.gz')
                session.get(p, 'T1km_crop.nii', 'freesurfer/wmparc_atlas.nii.gz')
                session.get(p, 'T2_crop.nii',   'freesurfer/wmparc_atlas.nii.gz')

        self.dataset.update(
            session.load_volumes(modality='MRI', id_prefix='nako')
        )

    # ------------------------------------------------------------------
    # NPZ export
    # ------------------------------------------------------------------

    def to_npz(self, output_path: str):
        """Resample all volumes to 1 mm isotropic spacing and save as .npz."""
        resample_and_save(self.dataset, output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NAKO (GPUnet) → .npz converter")
    parser.add_argument("--mode",    default="training_body",
                        choices=["training_body", "training_head",
                                 "training_combined", "testing_body", "testing_head"])
    parser.add_argument("--output",  default=None)
    parser.add_argument("--max-img", type=int, default=1000)
    args = parser.parse_args()

    loader = Nako_to_npz(mode=args.mode, max_img=args.max_img)
    loader.to_npz(args.output or f"GPUnet_{args.mode}")
