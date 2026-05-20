"""
evaluation/benchmark_nninteractive/nninteractive_inference.py
=============================================================
Clean wrapper around the nnInteractive inference session.

Metric fairness
---------------
Both ``vol_dice`` and ``window_dice`` are restricted to the axial range where
the ground-truth binary volume has foreground (the GT bounding box along
``prompt_axis``).  This makes these metrics directly comparable to Prompt U-Net,
which naturally only processes slices within the ROI extent.  The unrestricted
full-volume Dice is also returned as ``vol_dice_full`` for reference.

nnInteractive lives at:
    evaluation/benchmark_models/nnInteractive/

Its Python package is imported by adding that directory to sys.path.

Usage
-----
    from evaluation.benchmark_nninteractive.nninteractive_inference import NNInteractiveInference
    import torch

    nn = NNInteractiveInference(
        model_dir  = "path/to/nnInteractive_v1.0",
        device     = torch.device("cuda:0"),
    )

    result = nn.run(
        img_4d             = img_np,           # (1, X, Y, Z)  float32
        seg_3d             = gt_binary,        # (X, Y, Z)     int / float
        initial_prompt_3d  = prompt_3d_binary, # (X, Y, Z)
        user_interacts_idx = [],               # from InteractiveFeedbackLoop
        prompt_axis        = 0,
        prompt_idx         = 42,
    )
    print(result["vol_dice"], result["window_dice"])

Notes
-----
* Call ``initialize()`` explicitly if you need lazy loading, otherwise the
  constructor downloads/loads the model immediately.
* ``reset()`` clears all nnInteractive session state so the same
  NNInteractiveInference instance can be reused across multiple volumes.

Interaction design
------------------
IFL correction slices (``user_interacts_idx``) are combined with the initial
prompt into a single ``add_initial_seg_interaction`` call.  This places all
corrections in the initial-seg channel (-7), which is designed for full
segmentation masks and triggers a full-volume refinement pass.  The scribble
channels (-2/-1) are designed for sparse user annotations — passing dense
slice masks through ``add_scribble_interaction`` produces degraded results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Resolve nnInteractive package path
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent          # evaluation/benchmark_nninteractive/
_PROJECT_ROOT = _HERE.parent.parent                      # prompt-unet/
_NN_PKG_DIR   = _PROJECT_ROOT / "evaluation" / "benchmark_models" / "nnInteractive"

if str(_NN_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_NN_PKG_DIR))

# Also ensure the project root is reachable for utils imports
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Deferred heavy imports (torch / nnInteractive not needed at import time)
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for NNInteractiveInference. "
            "Install it with: pip install torch"
        )


def _require_session_cls():
    try:
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
        return nnInteractiveInferenceSession
    except ImportError as e:
        raise ImportError(
            f"Could not import nnInteractiveInferenceSession.  "
            f"Make sure the nnInteractive repo is present at:\n"
            f"  {_NN_PKG_DIR}\n"
            f"Original error: {e}"
        )


# ---------------------------------------------------------------------------
# NNInteractiveInference
# ---------------------------------------------------------------------------

class NNInteractiveInference:
    """
    Wraps an ``nnInteractiveInferenceSession`` for repeated volume inference.

    Parameters
    ----------
    model_dir : str or Path
        Path to the previously downloaded model weights directory
        (e.g. /some/path/nnInteractive_v1.0).
        If None, weights are downloaded automatically from HuggingFace Hub
        into *download_dir* (requires ``huggingface_hub``).
    device : torch.device or str
        Device for nnInteractive inference.  Defaults to CUDA:0 if available,
        else CPU.
    use_torch_compile : bool
        Experimental — not tested by default.
    n_threads : int or None
        CPU threads for nnInteractive.  Defaults to ``os.cpu_count()``.
    do_autozoom : bool
        Enable nnInteractive's AutoZoom for better patch coverage.
    use_pinned_memory : bool
        Optimises GPU memory transfers.  Keep False unless you have profiled
        that the transfer is a bottleneck.
    download_dir : str or Path
        Only used when *model_dir* is None.  Weights are stored here.
    repo_id : str
        HuggingFace Repo ID.  Default "nnInteractive/nnInteractive".
    model_name : str
        Model variant name inside the repo.  Default "nnInteractive_v1.0".
    verbose : bool
        Pass through to the nnInteractive session.
    """

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        device=None,
        use_torch_compile: bool = False,
        n_threads: Optional[int] = None,
        do_autozoom: bool = True,
        use_pinned_memory: bool = False,
        download_dir: str | Path = "/tmp/nnInteractive_weights",
        repo_id: str = "nnInteractive/nnInteractive",
        model_name: str = "nnInteractive_v1.0",
        verbose: bool = False,
    ):
        torch = _require_torch()

        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
            
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[nnInteractive] WARNING: CUDA requested but not found. Falling back to CPU.", file=sys.stderr)
            device = torch.device("cpu")

        self.device      = device
        self.model_name  = model_name

        # --- Resolve model directory ---
        if model_dir is None:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to auto-download nnInteractive weights. "
                    "pip install huggingface_hub  or supply model_dir manually."
                )
            print(f"[NNInteractiveInference] Downloading {model_name} from {repo_id} …")
            download_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"{model_name}/*"],
                local_dir=str(download_dir),
            )
            model_dir = Path(download_dir) / model_name
        else:
            model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"nnInteractive model directory not found: {model_dir}"
            )

        # --- Build session ---
        SessionCls = _require_session_cls()

        print(f"[NNInteractiveInference] Initialising session on device={device} …")
        self._session = SessionCls(
            device            = device,
            use_torch_compile = use_torch_compile,
            verbose           = verbose,
            torch_n_threads   = n_threads or os.cpu_count(),
            do_autozoom       = do_autozoom,
            use_pinned_memory = use_pinned_memory,
        )
        self._session.initialize_from_trained_model_folder(str(model_dir))
        print("[NNInteractiveInference] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear session state between volumes / runs."""
        self._session.reset_interactions()

    def run(
        self,
        img_4d: np.ndarray,
        seg_3d: np.ndarray,
        initial_prompt_3d: np.ndarray,
        user_interacts_idx: List[int],
        prompt_axis: int,
        prompt_idx: int,
        window: int = 10,
    ) -> dict:
        """
        Run nnInteractive on one volume.

        Interaction protocol
        --------------------
        The initial prompt and all IFL correction slices are combined into a
        single ``add_initial_seg_interaction`` call.  This writes the merged
        prompt to the initial-seg channel (-7), which is designed for full
        segmentation masks, and triggers a full-volume refinement pass.
        Using scribble channels (-2/-1) for full-slice GT masks is avoided
        because those channels are designed for sparse user annotations and
        produce degraded results with dense slice masks.

        Parameters
        ----------
        img_4d : np.ndarray, shape (1, X, Y, Z)
            Raw image volume with a leading channel dimension, as required by
            nnInteractive.  Values should be in the modality's native units
            (HU for CT, raw intensity for MRI) — nnInteractive normalises
            internally.
        seg_3d : np.ndarray, shape (X, Y, Z)
            Binary ground-truth volume.  Used for metric computation and
            to build the per-slice correction masks.
        initial_prompt_3d : np.ndarray, shape (X, Y, Z)
            3-D binary prompt from the initial slice interaction.
        user_interacts_idx : list[int]
            Slice indices where InteractiveFeedbackLoop substituted GT.
            These slices are combined with the initial prompt into a single
            ``add_initial_seg_interaction`` call.
        prompt_axis : int
        prompt_idx  : int
        window : int
            Half-width for the windowed Dice evaluation.

        Returns
        -------
        dict with keys:
            'vol_dice'     : float — volumetric Dice restricted to the GT-extent
                             (slices where seg_3d has foreground along prompt_axis).
                             Directly comparable to P-UNet's vol_dice.
            'vol_dice_full': float — volumetric Dice over the full 3-D volume
                             (kept for reference / regression checks).
            'window_dice'  : float — mean slice Dice in ±window around prompt,
                             also restricted to the GT extent.
            'result_volume': np.ndarray (X, Y, Z) uint8 — raw nnInteractive output.
        """
        import torch
        from utils.metrics import volumetric_dice, dice_window_nn

        if img_4d.ndim != 4:
            raise ValueError(
                f"img_4d must be 4-D with shape (1, X, Y, Z), got {img_4d.shape}"
            )

        initial_prompt_3d = initial_prompt_3d.astype(np.int16)

        # --- Build combined prompt (initial + any IFL correction slices) ---
        # Combining all corrections into the initial segmentation gives
        # nnInteractive the best possible refinement, since the initial-seg
        # channel (-7) is designed for full segmentation masks, unlike the
        # scribble channels (-2/-1) which are meant for sparse annotations.
        if user_interacts_idx:
            combined_prompt = initial_prompt_3d.copy()
            for idx in user_interacts_idx:
                interact_slice = np.take(seg_3d, idx, axis=prompt_axis)
                interact_3d    = np.zeros(img_4d.shape[1:], dtype=np.int16)
                if prompt_axis == 0:
                    interact_3d[idx, :, :]  = interact_slice
                elif prompt_axis == 1:
                    interact_3d[:, idx, :]  = interact_slice
                else:
                    interact_3d[:, :, idx]  = interact_slice
                np.maximum(combined_prompt, interact_3d, out=combined_prompt)
        else:
            combined_prompt = initial_prompt_3d

        # --- nnInteractive session ---
        self._session.set_image(img_4d)

        target_tensor = torch.zeros(tuple(img_4d.shape[1:]), dtype=torch.uint8)
        self._session.set_target_buffer(target_tensor)

        with torch.no_grad():
            try:
                # Single initial-seg interaction with all prompt information.
                # add_initial_seg_interaction resets all interactions,
                # writes the combined prompt to channel -7, and runs a
                # full-volume refinement pass (force_full_refine=True).
                self._session.add_initial_seg_interaction(
                    combined_prompt, run_prediction=True
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[NNInteractiveInference] GPU OOM — clearing cache and retrying …")
                    torch.cuda.empty_cache()
                    raise

        # --- Retrieve & convert ---
        result_np = np.asarray(target_tensor.clone(), dtype=np.int16)

        # --- Dice restricted to the GT extent (fair comparison with P-UNet) ---
        # P-UNet naturally only visits non-empty GT slices, so its vol_dice and
        # window_dice never include background slices.  We apply the same crop
        # here so the numbers are on the same footing.
        sum_axes      = tuple(a for a in range(seg_3d.ndim) if a != prompt_axis)
        gt_presence   = seg_3d.sum(axis=sum_axes) > 0   # True for each slice with foreground
        gt_slice_idxs = np.where(gt_presence)[0]

        vol_dice_full = volumetric_dice(seg_3d, result_np)   # full volume — reference only

        if len(gt_slice_idxs) == 0:
            # Degenerate: GT is empty — both metrics are trivially 1.0 / 1.0
            vol_dice    = vol_dice_full
            window_dice = vol_dice_full
        else:
            lo, hi = int(gt_slice_idxs[0]), int(gt_slice_idxs[-1]) + 1   # [lo, hi)
            sl = [slice(None)] * seg_3d.ndim
            sl[prompt_axis] = slice(lo, hi)
            sl = tuple(sl)

            vol_dice = volumetric_dice(seg_3d[sl], result_np[sl])

            # Adjust the prompt index to be relative to the cropped volume
            adj_prompt_idx = max(0, min(prompt_idx - lo, hi - lo - 1))
            window_dice = dice_window_nn(
                seg_3d[sl], result_np[sl], prompt_axis, adj_prompt_idx, window=window
            )

        self._session.reset_interactions()

        return {
            "vol_dice"     : vol_dice,       # GT-extent crop
            "vol_dice_full": vol_dice_full,  # full volume (reference)
            "window_dice"  : window_dice,    # GT-extent crop
            "result_volume": result_np,
        }
