"""
dpx_loader.py  —  DPX / patchwork server loading utilities
===========================================================

Shared infrastructure for all *_to_npz converters in this folder.

╔══════════════════════════════════════════════════════════════════════════════╗
║  SERVER-SPECIFIC CODE — NOT PORTABLE                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  This module depends on:                                                     ║
║    • patchwork_dev  — an internal research library                           ║
║    • DPX_core       — a proprietary data-management framework                ║
║    • Environment variables DPXROOT and DPXproject                            ║
║                                                                              ║
║  None of these are publicly available.  The DPX platform is specific to     ║
║  the institution where the training data is stored.                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  HOW TO REPLACE THIS WITH YOUR OWN DATA                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  You are free to replace DPXSession entirely.  The only contract each       ║
║  *_to_npz script expects is that self.dataset is populated with entries     ║
║  of this form:                                                               ║
║                                                                              ║
║      dataset[pid] = {                                                        ║
║          "image":         np.ndarray | tf.Tensor,  # shape (Z, Y, X)        ║
║          "segmentations": np.ndarray | tf.Tensor,  # shape (Z, Y, X)        ║
║          "modality":      "CT" | "MRI",                                     ║
║          "spacing":       (sz, sy, sx),            # voxel size in mm       ║
║      }                                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import numpy as np
import tensorflow as tf


# ============================================================================
# Shared utilities  (used by all converters)
# ============================================================================

def get_spacing(r) -> tuple:
    """
    Extract voxel spacing (sz, sy, sx) in mm from a patchwork rset_ entry.

    Handles the most common return formats:
    - Dictionary: {'voxsize': [x, y, z], ...} (returned by modern patchwork)
    - Array/Tensor: [z, y, x] (returned by some legacy versions)

    Falls back to (1.0, 1.0, 1.0) if the format is unrecognised.

    Note
    ----
    If a dictionary with 'voxsize' is found, the elements [x, y, z] are
    reversed to return (z, y, x) in mm, matching the pipeline's expectations.
    """
    try:
        # 1. Handle modern patchwork dictionary format
        if isinstance(r, dict) and 'voxsize' in r:
            r = r['voxsize']
            arr = np.asarray(r, dtype=float).flatten()
            if arr.size >= 3:
                # [x, y, z] -> (z, y, x)
                return (float(arr[2]), float(arr[1]), float(arr[0]))

        # 2. Handle legacy array / tensor formats
        if tf.is_tensor(r):
            r = r.numpy()
        arr = np.asarray(r, dtype=float).flatten()
        if arr.size >= 3:
            # Assume [z, y, x] as per original loader convention
            return tuple(float(v) for v in arr[:3])
        if arr.size == 1:
            v = float(arr[0])
            return (v, v, v)
    except Exception:
        pass
    print("WARNING: Could not extract voxel spacing from rset_ entry; "
          "assuming 1 mm isotropic.")
    return (1.0, 1.0, 1.0)


def resample_and_save(dataset: dict, output_path: str) -> None:
    """
    Resample every entry in *dataset* to 1 mm isotropic spacing and write
    a compressed .npz archive via ds_handler.

    This is the shared final step for all *_to_npz converters.  It is NOT
    used by HanSeg_to_npz.py, which reads voxel spacing directly from the
    SimpleITK image object and folds resampling into its per-patient loop.

    Parameters
    ----------
    dataset : dict
        ``{pid: {"image": array/tensor (Z,Y,X),
                 "segmentations": array/tensor (Z,Y,X),
                 "modality": "CT"|"MRI",
                 "spacing": (sz, sy, sx)}}``
        Produced by DPXSession.load_volumes() or any compatible loader.
    output_path : str
        Destination file path (with or without ``.npz`` extension).
    """
    from pathlib import Path
    _root = str(Path(__file__).resolve().parent.parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from data.test_data.ds_handler import save_dataset
    from utils.resampling import resample_isotropic

    out = {}
    print(f"\nResampling {len(dataset)} entries to 1 mm isotropic ...")

    for pid, item in dataset.items():
        img = item["image"]
        seg = item["segmentations"]
        if tf.is_tensor(img): img = img.numpy()
        if tf.is_tensor(seg): seg = seg.numpy()
        img = np.asarray(img, dtype=np.float32)
        seg = np.asarray(seg, dtype=np.float32)

        spacing  = item.get("spacing",  (1.0, 1.0, 1.0))
        modality = item.get("modality", "UNKNOWN")
        is_ct    = (modality == "CT")

        print(f"  [{pid}]  modality={modality}  shape={img.shape}  "
              f"spacing={tuple(f'{s:.2f}' for s in spacing)} mm")

        img_iso = resample_isotropic(img, spacing, is_mask=False, is_ct=is_ct)
        seg_iso = resample_isotropic(seg, spacing, is_mask=True,  is_ct=False)

        out[str(pid)] = {
            "image":         img_iso,
            "segmentations": seg_iso.astype(np.uint8),
            "modality":      modality,
        }

    save_dataset(out, output_path)
    print(f"Saved {len(out)} entries → {output_path}.npz")


# ============================================================================
# DPX session
# ============================================================================

class DPXSession:
    """
    Manages a single DPX data-loading session.

    Typical usage
    -------------
    ::

        session = DPXSession()

        # Accumulate one or more patient / STAG selectors:
        session.get('STAG:train_ct', 'ct.nii.gz', 'labels.nii.gz')

        # Trigger patchwork loading and get a ready-to-use dict:
        dataset = session.load_volumes(max_img=500, modality='CT', id_prefix='ct')

        # For a second modality, reset and reuse:
        session.reset()
        session.get('STAG:train_mri', 'img.nii.gz', 'labels.nii.gz')
        dataset.update(session.load_volumes(max_img=500, modality='MRI', id_prefix='mri'))

    Parameters / calling notes
    --------------------------
    *max_img* in load_volumes() limits the number of *patients* (not entries).
    Pass it when using STAG selectors that match an unbounded set.  For
    explicit patient-ID loops (NAKO) the limit is enforced in the calling
    code, so max_img=None is appropriate.

    id_prefix prevents key collisions when multiple load_volumes() calls
    write into the same dataset dict (e.g. 'ct' vs 'mri' for MSD).
    """

    def __init__(self):
        self._ximg: dict = {}
        self._limg: dict = {}
        self._pw         = None   # patchwork module
        self._dpx        = None   # DPX_selectFiles function
        self._project    = None
        self._init_env()

    # ------------------------------------------------------------------

    def _init_env(self):
        """Bootstrap the DPX / patchwork environment (idempotent)."""
        sys.path.append("/software")
        sys.path.append(os.environ['DPXROOT'] + '/src/python')

        import patchwork_dev.patchwork as patchwork
        from DPX_core import DPX_selectFiles

        self._pw      = patchwork
        self._dpx     = DPX_selectFiles
        self._project = os.environ['DPXproject']

    # ------------------------------------------------------------------

    def get(self, ids: str, img: str, label: str):
        """
        Accumulate an image / label pair into the session.

        Parameters
        ----------
        ids   : DPX patient ID or STAG selector,
                e.g. ``'104171'`` or ``'STAG:train_ct'``.
        img   : image filename, e.g. ``'ct.nii.gz'``.
        label : label filename, e.g. ``'labels.nii.gz'``.
        """
        ext = lambda d, e: {(k + e): d[k] for k in d}

        # Fetch separately so we can log what the server returns
        res_img   = self._dpx(self._project, [ids, img])
        res_label = self._dpx(self._project, [ids, label])

        print(f"  [DPX] get({ids!r:30s}, {img!r}) "
              f"\u2192 img:{len(res_img)} label:{len(res_label)} files")

        if not res_img:
            print(f"  [DPX] WARNING: no image files matched {ids!r} / {img!r} "
                  f"\u2014 check STAG name, patient ID, and file path.")
        if not res_label:
            print(f"  [DPX] WARNING: no label files matched {ids!r} / {label!r} "
                  f"\u2014 check label path.")

        # Both dicts are keyed by the *image* name (original patchwork convention).
        self._ximg.update(ext(res_img,   img))
        self._limg.update(ext(res_label, img))


    # ------------------------------------------------------------------

    def reset(self):
        """
        Clear all accumulated selections so the session can be reused
        for a second modality group without creating a new DPXSession.
        """
        self._ximg, self._limg = {}, {}

    # ------------------------------------------------------------------

    def load_volumes(self,
                     max_img: int = None,
                     modality: str = 'CT',
                     id_prefix: str = '') -> dict:
        """
        Trigger the patchwork loading pipeline and return a dataset dict.

        Parameters
        ----------
        max_img    : Maximum number of *patients* to include.  ``None`` means
                     no limit.  Use when loading via STAG selectors.
        modality   : ``'CT'`` or ``'MRI'`` — stored in every returned entry.
        id_prefix  : Prepended to every patient ID, e.g. ``'ct'`` or ``'mri'``.

        Returns
        -------
        dict[str, dict]  with entries::

            {pid: {
                "image":         tf.Tensor (Z, Y, X),
                "segmentations": tf.Tensor (Z, Y, X),  # integer labels ≥ 0
                "modality":      str,
                "spacing":       (sz, sy, sx),          # mm
            }}
        """
        if not self._ximg:
            print("WARNING: load_volumes() called with no selections — "
                  "call get() first.")
            return {}

        getfname = lambda d: [{k: d[k][list(d[k].keys())[0]]['FilePath'] for k in d}]
        # No tf.device context — for offline NPZ generation there are no TF
        # training ops that require CPU pinning, and the context serialises
        # patchwork's internal loading threads (causing the slowdown).
        loading   = {"integer_labels": True}
        contrasts = getfname(self._ximg)
        labels    = getfname(self._limg)
        subjects  = list(contrasts[0].keys())

        tset_, lset_, rset_, subjs = self._pw.improc_utils.load_data_structured(
            contrasts=contrasts, labels=labels, subjects=subjects, **loading
        )

        tset, lset, rset = [], [], []

        for k in range(len(tset_)):
            if max_img is not None and k >= max_img:
                break
            for j in range(tset_[k].shape[4]):
                tset.append(tset_[k][..., j:j + 1])
                tmp_ = lset_[k][:, :, :, :, 0:1]
                tmp_ = tf.where(tmp_ < 0, 0, tmp_)
                lset.append(tf.cast(tmp_, dtype=tf.int32))
                rset.append(rset_[k])

        result: dict = {}

        for idx in range(len(tset)):
            seq   = idx + 1
            pid   = f"{id_prefix}_{seq}" if id_prefix else str(seq)
            result[pid] = {
                "image":         tf.squeeze(tset[idx]),
                "segmentations": tf.squeeze(lset[idx]),
                "modality":      modality,
                "spacing":       get_spacing(rset[idx]),
            }

        return result

