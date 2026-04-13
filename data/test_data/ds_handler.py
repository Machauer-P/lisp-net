"""
ds_handler.py
===========
Save / load dataset dicts as compressed NumPy archives (.npz).

Dataset dict format:
    {pid: {
        "image":        np.ndarray,                      # 3-D volume (Z, Y, X)
        "segmentations": np.ndarray | list[np.ndarray],  # single or per-organ masks
        "modality":     str,                             # e.g. "CT", "MRI", "OTHER"
    }}

Internal .npz layout
--------------------
    _pids           – 1-D string array of patient IDs
    _modalities     – 1-D string array of modality tags (parallel to _pids)
    _seg_counts     – 1-D int array, number of seg arrays per patient
    _seg_is_list    – 1-D bool array, True if segmentations was originally a list
    {i}_image       – image volume for patient index i
    {i}_seg_{j}     – j-th segmentation array for patient index i

Note: Every .npz produced by this pipeline stores volumes that have already been
resampled to 1 mm isotropic spacing by resample_isotropic().  Do *not* resample
again during training.
"""

import os
import numpy as np


def save_dataset(dataset: dict, filename: str) -> str:
    """
    Save a dataset dict as a compressed .npz file.

    Parameters
    ----------
    dataset : dict
        Keys are patient IDs (str), values are dicts with:
          - "image":        np.ndarray (3-D volume, already isotropically resampled)
          - "segmentations": np.ndarray or list of np.ndarray
          - "modality":     str — imaging modality tag, e.g. "CT", "MRI", "OTHER"
    filename : str
        Output path. ".npz" is appended automatically if not present.

    Returns
    -------
    str – the final output path.
    """
    out = filename if filename.endswith(".npz") else filename + ".npz"

    arrays: dict[str, np.ndarray] = {}
    pids: list[str]  = []
    modalities: list[str] = []
    seg_counts: list[int] = []
    seg_is_list: list[bool] = []

    for i, (pid, item) in enumerate(dataset.items()):
        pids.append(str(pid))
        modalities.append(str(item.get("modality", "UNKNOWN")))
        arrays[f"{i}_image"] = np.asarray(item["image"])

        segs = item.get("segmentations", item.get("segmentation"))

        if isinstance(segs, (list, tuple)):
            seg_is_list.append(True)
            for j, seg in enumerate(segs):
                arrays[f"{i}_seg_{j}"] = np.asarray(seg)
            seg_counts.append(len(segs))
        else:
            seg_is_list.append(False)
            arrays[f"{i}_seg_0"] = np.asarray(segs)
            seg_counts.append(1)

    # Metadata arrays
    arrays["_pids"]        = np.array(pids,       dtype=str)
    arrays["_modalities"]  = np.array(modalities, dtype=str)
    arrays["_seg_counts"]  = np.array(seg_counts, dtype=np.int32)
    arrays["_seg_is_list"] = np.array(seg_is_list, dtype=bool)

    dirname = os.path.dirname(out)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    print(f"\nSaving dataset to {out} ...")
    np.savez_compressed(out, **arrays)
    print("Save complete!")
    return out


def load_dataset(filename: str) -> dict:
    """
    Load a dataset dict from a .npz file.

    Returns the same dict structure that was saved:
        {pid: {
            "image":        np.ndarray,
            "segmentations": np.ndarray | list[np.ndarray],
            "modality":     str,
        }}
    """
    data = np.load(filename, allow_pickle=False)

    pids       = data["_pids"]
    seg_counts = data["_seg_counts"]
    modalities = data["_modalities"] if "_modalities" in data else None

    # _seg_is_list may be absent in manually-created files
    seg_is_list = data["_seg_is_list"] if "_seg_is_list" in data else None

    dataset: dict = {}
    for i, pid in enumerate(pids):
        pid = str(pid)
        image = data[f"{i}_image"]

        seg_count  = int(seg_counts[i])
        seg_arrays = [data[f"{i}_seg_{j}"] for j in range(seg_count)]

        # Reconstruct original segmentation structure
        if seg_is_list is not None and not seg_is_list[i] and seg_count == 1:
            segs = seg_arrays[0]   # was originally a single array
        else:
            segs = seg_arrays      # was originally a list

        modality = str(modalities[i]) if modalities is not None else "UNKNOWN"

        dataset[pid] = {
            "image":         image,
            "segmentations": segs,
            "modality":      modality,
        }

    return dataset
