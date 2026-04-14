import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_one_dp(x, y, p, offset, contrast=1):
    fig, axes = plt.subplots(1, 2, figsize=(7, 7))

    y = np.squeeze(y.numpy())
    x = np.squeeze(x.numpy())
    p = np.squeeze(p.numpy())

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
    fig, axes = plt.subplots(1, 3, figsize=(7, 7))

    y = np.squeeze(y.numpy())
    x = np.squeeze(x.numpy())

    # --- Query + Target (gelb) ---
    y_mask = (y > 0)
    y_img = np.stack([x, x, x], axis=-1)
    y_img[y_mask] = [1, 1, 0]  # gelb

    axes[0].imshow(y_img)
    axes[0].set_title('Query (x) + Target (y)', fontsize=10)
    axes[0].axis("off")

    # --- Prompt (gelb) ---
    p = np.squeeze(p.numpy())
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

def visualize_a_few_results(model_name:str, loaded_model: tf.keras.Model, ds, offset, img_to_plot=8, threshold=0.45, contrast=1):
    from utils.metrics import dice_score_tf
    for i, (x,y,p) in enumerate(ds):
        if i == img_to_plot:
            break
            
        x = tf.expand_dims(x, axis=0) 
        p = tf.expand_dims(p, axis=0)
        pred = loaded_model.predict([x[0:1,:,:,0:1], p[0:1,...]])
        pred = tf.where(pred < threshold, 0.0, pred)
        pred = tf.where(pred >= threshold, 1.0, pred)

        plot_result(x,y,p,pred,offset[i],f'Prediction (Number {str(i)})', contrast)
        y = tf.cast(y, tf.float32)
        print(f"Dice: {dice_score_tf(y[..., 0:1], pred):.3f}\n")

def plot_random_slice_from_vol(dataset, num_examples=9):
    """
    Takes a tf.data.Dataset of shape (num_slices, height, width)
    and plots one random slice from each Dataset.

    Args:
        dataset: tf.data.Dataset
        num_examples: number of dataset elements to sample and plot
    """
    plt.figure(figsize=(10, 10))

    for i, (x, y) in enumerate(dataset.take(num_examples)):
        # volume shape: (num_slices, height, width)
        num_slices = tf.shape(x)[0]

        # Choose a random slice index
        rand_idx = tf.random.uniform([], minval=0, maxval=num_slices, dtype=tf.int32)
        slice_img = x[rand_idx, :, :]
        slice_segm = y[rand_idx, :, :]

        # Convert to numpy for plotting
        slice_img = slice_img.numpy()
        slice_segm = slice_segm.numpy()

        plt.subplot(int(np.ceil(np.sqrt(num_examples))), int(np.ceil(np.sqrt(num_examples))), i + 1)
        plt.imshow(slice_img + slice_segm)
        plt.axis('off')
        plt.title(f"Slice {rand_idx.numpy()}")

    plt.tight_layout()
    plt.show()

def plot_samples_from_vol(dataset, idx_list, num_img=10, max_entries=300):
    """
    Plots `num_img` images evenly spaced from a TensorFlow dataset of (image, label) pairs or a tuple of (vol_img, vol_labels).

    Args:
        dataset (tf.data.Dataset): Dataset containing (image, label) pairs or a tuple of (vol_img, vol_labels).
        idx_list (list): List with the index of the current slice. Needs to match with dataset. 
        num_img (int): Number of images to plot.
        max_entries (int): Limit to this many samples for faster plotting.
    """
    if type(dataset) == tf.data.Dataset:

        count = sum(1 for _ in dataset.take(max_entries))
        if count == 0:
            print("Dataset is empty.")
            return

        if count < num_img:
            num_img = count

        indices = [int(i * count / num_img) for i in range(num_img)]

        plt.figure(figsize=(num_img * 3, 3))
        for idx, (image, label) in enumerate(dataset.take(max_entries)):
            if idx in indices:
                plot_idx = indices.index(idx) + 1
                plt.subplot(1, num_img, plot_idx)
                plt.imshow(image.numpy().squeeze() + label.numpy().squeeze())
                plt.title(str(idx_list[idx]), fontsize=14)
                plt.axis("off")

        plt.tight_layout()
        plt.show()

    elif type(dataset) == tuple:

        vol, labels = dataset

        count = vol.shape[0] 

        if count < num_img:
            num_img = count

        indices = [int(i * vol.shape[0] / num_img) for i in range(num_img)]

        plt.figure(figsize=(num_img * 3, 3))
        # First dim == img counter
        for i in indices:
            image = vol[i,...]
            label = labels[i,...]

            plot_idx = indices.index(i) + 1
            plt.subplot(1, num_img, plot_idx)
            plt.imshow(image.numpy().squeeze() + label.numpy().squeeze())
            plt.title(str(idx_list[i]), fontsize=14)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    else:
        return
