import numpy as np
import matplotlib.pyplot as plt

def plot_one_dp(x, y, p, offset, contrast=1):
    """
    Plots a single data point consisting of the image (x), ground truth mask (y), and the prompt (p).
    
    Args:
        x (tf.Tensor/np.ndarray): The input image slice.
        y (tf.Tensor/np.ndarray): The target segmentation mask.
        p (tf.Tensor/np.ndarray): The prompt tensor (image + prompt channel).
        offset (int): The offset value used for the prompt, to be displayed in the title.
        contrast (float, optional): Contrast adjustment factor. Defaults to 1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 7))

    y = np.squeeze(np.asarray(y))
    x = np.squeeze(np.asarray(x))
    p = np.squeeze(np.asarray(p))

    # Binär/Contrast setzen
    y_mask = (y > 0)
    y_img = np.stack([x, x, x], axis=-1)  # RGB
    y_img[y_mask] = [1, 1, 0]  # Gelb

    axes[0].imshow(y_img)
    axes[0].set_title('Query (x) + Target (y)')
    axes[0].axis("off") 

    # Prompt visualisieren
    p1 = p[..., 1]
    p1_mask = (p1 > 0)

    p_img = np.stack([p[..., 0], p[..., 0], p[..., 0]], axis=-1)
    p_img[p1_mask] = [1, 1, 0]  # Gelb

    axes[1].imshow(p_img)
    axes[1].set_title(f'Prompt (offset = {offset})')
    axes[1].axis("off") 

    plt.show()
    print()

def plot_result(x, y, p, pred, offset, pred_titel='', contrast=1, show=True):
    """
    Plots the input image with ground truth, the prompt, and the model's prediction side-by-side.
    
    Args:
        x (tf.Tensor): Input image.
        y (tf.Tensor): Ground truth mask.
        p (tf.Tensor): Prompt tensor.
        pred (np.ndarray): Model prediction mask.
        offset (int): Prompt offset for title display.
        pred_titel (str, optional): Title for the prediction plot. Defaults to ''.
        contrast (float, optional): Contrast adjustment factor. Defaults to 1.
        show (bool, optional): Whether to call plt.show(). Defaults to True.
        
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7, 7))

    y = np.squeeze(np.asarray(y))
    x = np.squeeze(np.asarray(x))

    # --- Query + Target (gelb) ---
    y_mask = (y > 0)
    y_img = np.stack([x, x, x], axis=-1)
    y_img[y_mask] = [1, 1, 0]  # gelb

    axes[0].imshow(y_img)
    axes[0].set_title('Query (x) + Target (y)', fontsize=10)
    axes[0].axis("off")

    # --- Prompt (gelb) ---
    p = np.squeeze(np.asarray(p))
    p1 = p[..., 1]
    p1_mask = (p1 > 0)

    p_img = np.stack([p[..., 0], p[..., 0], p[..., 0]], axis=-1)
    p_img[p1_mask] = [1, 1, 0]  # gelb

    axes[1].imshow(p_img)
    axes[1].set_title(f'Prompt (offset = {offset})', fontsize=10)
    axes[1].axis("off")

    # --- Prediction ---
    axes[2].imshow(np.squeeze(pred))
    axes[2].set_title(pred_titel, fontsize=10)
    axes[2].axis("off")

    if show:
        plt.show()
    return fig

def visualize_a_few_results(model_name: str, loaded_model, ds, offset, img_to_plot=8, threshold=0.45, contrast=1):
    """
    Runs prediction on a few samples from a dataset and visualizes the results.

    Args:
        model_name (str): Name of the model (for display/logging).
        loaded_model: The trained Keras model.
        ds: Dataset containing (x, y, p) tuples (tf.data.Dataset or iterable).
        offset (list/np.ndarray): List of offsets corresponding to the dataset samples.
        img_to_plot (int, optional): Number of samples to visualize. Defaults to 8.
        threshold (float, optional): Binary threshold for the prediction mask. Defaults to 0.45.
        contrast (float, optional): Contrast adjustment for plots. Defaults to 1.
    """
    from utils.metrics import dice_score_tf
    for i, (x, y, p) in enumerate(ds):
        if i == img_to_plot:
            break

        x_np = np.asarray(x)
        p_np = np.asarray(p)

        # Add batch dim if needed
        if x_np.ndim == 3:
            x_np = x_np[np.newaxis]
        if p_np.ndim == 3:
            p_np = p_np[np.newaxis]

        pred = loaded_model.predict([x_np[0:1, :, :, 0:1], p_np[0:1]])
        pred = np.where(pred < threshold, 0.0, 1.0)

        plot_result(x, y, p, pred, offset[i], f'Prediction (Number {str(i)})', contrast)
        y_np = np.asarray(y).astype(np.float32)
        print(f"Dice: {dice_score_tf(y_np[..., 0:1], pred):.3f}\n")

def visualize_bundle(path_to_npz):
    """
    Visualizes the support set and a few query samples from a generated 2D NPZ bundle.
    
    Args:
        path_to_npz (str): Path to the saved .npz bundle file.
    """
    from data.test_data.ds_handler_2d import load_2d_npz_bundle
    from pathlib import Path
    
    query, support = load_2d_npz_bundle(path_to_npz)
    
    print(f"Loaded NPZ bundle: {Path(path_to_npz).name}")
    print(f"Support Set: sx={support['sx'].shape}, sy={support['sy'].shape}")
    print(f"Query Set:   x={query['x'].shape}, y={query['y'].shape}, p={query['p'].shape}")
    
    # --- 1. Plot a few Support Samples ---
    s_display = min(4, len(support['sx']))
    fig, axes = plt.subplots(2, s_display, figsize=(s_display * 3, 6))
    fig.suptitle("Support Set Samples", fontsize=16)
    
    for i in range(s_display):
        img = support['sx'][i, ..., 0]
        lbl = support['sy'][i, ..., 0]
        # Modality: 0=CT, 1=MRI
        mod_str = "MRI" if support['s_modality'][i] == 1.0 else "CT"
        
        ax_img = axes[0, i] if s_display > 1 else axes[0]
        ax_lbl = axes[1, i] if s_display > 1 else axes[1]
        
        ax_img.imshow(img, cmap='gray', vmin=-5, vmax=5)
        ax_img.set_title(f"Img [{mod_str}]")
        ax_img.axis('off')
        
        ax_lbl.imshow(lbl, cmap='gray', vmin=0, vmax=1)
        ax_lbl.set_title("Mask")
        ax_lbl.axis('off')
    plt.tight_layout()
    plt.show()
    
    # --- 2. Plot a few Query Samples and Prompts ---
    q_display = min(4, len(query['x']))
    fig, axes = plt.subplots(q_display, 4, figsize=(12, q_display * 3))
    fig.suptitle("Query Set Pairs", fontsize=16)
    
    for i in range(q_display):
        x_i = query['x'][i, ..., 0]
        y_i = query['y'][i, ..., 0]
        p_img = query['p'][i, ..., 0]
        p_lbl = query['p'][i, ..., 1]
        offset = query['offset'][i]
        mod_str = "MRI" if query['modality'][i] == 1.0 else "CT"
        
        ax_pimg = axes[i, 0] if q_display > 1 else axes[0]
        ax_plbl = axes[i, 1] if q_display > 1 else axes[1]
        ax_x    = axes[i, 2] if q_display > 1 else axes[2]
        ax_y    = axes[i, 3] if q_display > 1 else axes[3]
        
        ax_pimg.imshow(p_img, cmap='gray', vmin=-5, vmax=5)
        ax_pimg.set_title(f"Prompt Img [{mod_str}]\n(Offset: {offset})")
        ax_pimg.axis('off')
        
        ax_plbl.imshow(p_lbl, cmap='gray', vmin=0, vmax=1)
        ax_plbl.set_title("Prompt Mask")
        ax_plbl.axis('off')
        
        ax_x.imshow(x_i, cmap='gray', vmin=-5, vmax=5)
        ax_x.set_title("Target Img")
        ax_x.axis('off')
        
        ax_y.imshow(y_i, cmap='gray', vmin=0, vmax=1)
        ax_y.set_title("Target Mask")
        ax_y.axis('off')
        
    plt.tight_layout()
    plt.show()

def visualize_img_with_mask(img, mask, alpha=0.5):
    """
    Interactive volume visualizer using ipywidgets.
    Allows sliding through Z-slices of a 3D volume with an overlaid mask.
    
    NOTE: The interactive state and specific slice view are NOT persisted 
    in notebook outputs after a kernel restart or page reload.
    
    Args:
        img (np.ndarray): 3D image volume (Z, H, W).
        mask (np.ndarray): 3D segmentation mask (Z, H, W).
        alpha (float, optional): Transparency of the mask overlay. Defaults to 0.5.
    """
    import ipywidgets as widgets
    from IPython.display import display
    
    # Ensure correct shape (Z, H, W)
    if len(img.shape) == 4 and img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
    if len(mask.shape) == 4 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
        
    assert img.shape == mask.shape, f"Shape mismatch: {img.shape} vs {mask.shape}"

    depth = img.shape[0]

    def show_slice(idx):
        plt.figure(figsize=(6,6))
        plt.imshow(img[idx], cmap='gray')
        plt.imshow(mask[idx], cmap='jet', alpha=alpha)
        plt.title(f"Slice {idx}")
        plt.axis('off')
        plt.show()

    slider = widgets.IntSlider(
        value=depth//2,
        min=0,
        max=depth-1,
        step=1,
        description='Slice:'
    )

    widgets.interact(show_slice, idx=slider)

def plot_vol_slices(img, mask, num_slices=5, alpha=0.5, figsize=(15, 5)):
    """
    Static volume visualizer showing a horizontal grid of evenly spaced slices.
    
    Unlike visualize_img_with_mask, the output of this function IS persisted 
    in the notebook file, making it ideal for documenting results.
    
    Args:
        img (np.ndarray): 3D image volume (Z, H, W).
        mask (np.ndarray): 3D segmentation mask (Z, H, W).
        num_slices (int, optional): Number of slices to display. Defaults to 5.
        alpha (float, optional): Transparency of the mask overlay. Defaults to 0.5.
        figsize (tuple, optional): Size of the resulting figure. Defaults to (15, 5).
        
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    # Ensure correct shape (Z, H, W)
    if len(img.shape) == 4 and img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
    if len(mask.shape) == 4 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
        
    assert img.shape == mask.shape, f"Shape mismatch: {img.shape} vs {mask.shape}"
    
    depth = img.shape[0]
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(1, num_slices, figsize=figsize, squeeze=False)
        
    for i, idx in enumerate(indices):
        ax = axes[0, i]
        ax.imshow(img[idx], cmap='gray')
        ax.imshow(mask[idx], cmap='jet', alpha=alpha)
        ax.set_title(f"Slice {idx}")
        ax.axis('off')
        
    plt.tight_layout()
    return fig
