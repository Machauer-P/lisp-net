import numpy as np
from scipy.ndimage import zoom


def resample_isotropic(volume, voxel_sizes, is_mask=False, is_ct=False):
    """
    Resample a 3-D volume to isotropic 1 mm resolution.

    This function is intended for use at *data preparation time* (e.g. when
    converting raw NRRD/NIfTI files to .npz archives).  It operates entirely
    in NumPy so it can run outside of a TensorFlow graph or session.  Every
    .npz file produced by the pipeline stores volumes that have already been
    resampled to 1 mm isotropic spacing, so this function should *not* be
    called again inside the DataGenerator.

    Args:
        volume    : np.ndarray, shape (Z, Y, X) — the raw volume.
        voxel_sizes: tuple of floats (sz, sy, sx) — original voxel spacing in mm.
        is_mask   : bool — True when resampling a segmentation/label volume.
                    Uses nearest-neighbour interpolation to prevent label blurring.
        is_ct     : bool — True when the modality is Computed Tomography.
                    Pads new voxels with -1024 HU (air) instead of 0.

    Returns:
        np.ndarray (float32) resampled to 1 mm isotropic spacing.
    """
    target_spacing = 1.0

    # Zoom factors: original_spacing / target_spacing
    zoom_factors = [vs / target_spacing for vs in voxel_sizes]

    # Interpolation order:
    #   0 = Nearest Neighbour  →  masks  (no label blurring)
    #   1 = Trilinear          →  images (smooth intensity)
    order = 0 if is_mask else 1

    # Background padding value (cval):
    #   mask  → 0            (background class)
    #   CT    → -1024 HU     (air, the physical background for CT)
    #   MRI   → 0            (conventional black background)
    if is_mask:
        bg_value = 0
    elif is_ct:
        bg_value = -1024
    else:
        bg_value = 0

    volume_iso = zoom(
        volume,
        zoom=zoom_factors,
        order=order,
        mode='constant',
        cval=bg_value,
    )

    return volume_iso.astype(np.float32)


# def gen_grid(shape):
#     # slice_size = max(shape)
#     slice_size = int(np.ceil(np.linalg.norm(shape)))
#     x_grid = np.linspace(-slice_size//2, slice_size//2, slice_size)
#     y_grid = np.linspace(-slice_size//2, slice_size//2, slice_size)
#     grid_x, grid_y = np.meshgrid(x_grid, y_grid)
#     return grid_x, grid_y

# def __random_plane_coords(shape, center:np.array, center_offset:np.array, grid_x, grid_y):
#     """Erzeugt zufällige Schnittebene und liefert die 3D-Koordinaten für jeden Pixel.
#     """
#     # int(np.ceil(np.linalg.norm(shape))) macht Bild größer, deswegen max(shape). Es gibt das Risiko, dass etwas abgeschnitten wird, je nach Perspektive
#     # slice_size = max(shape)
#     slice_size = int(np.ceil(np.linalg.norm(shape)))  # Raumdiagonale

#     # Zufälliger Mittelpunkt in Voxelkoordinaten
#     # center = np.array([np.random.uniform(0, shape[0]),
#     #                    np.random.uniform(0, shape[1]),
#     #                    np.random.uniform(0, shape[2])])

#     # Zufällige Normale (Richtung der Ebene)
#     normal = np.random.randn(3)
#     normal /= np.linalg.norm(normal) # Einheitsvektor erzeugen

#     # Zwei Vektoren in der Ebene finden, um Ebene aufzuspannen
#     v1 = np.random.randn(3) # zufälliger Vektor
#     v1 -= v1.dot(normal) * normal # Anteil entlang der Normalen entfernen
#     v1 /= np.linalg.norm(v1) # Einheitsvektor
#     v2 = np.cross(normal, v1) # Senkrechter Vektor erzeugen
#     # v1 = x-Richtung und v2 = y-Richtung der Ebene
#     # Beide liegen in der Ebene, sind rechtwinklig und haben Länge 1

#     # Vektorisierte Koordinatenberechnung (kein for-loop!)
#     coords = np.zeros((3, slice_size, slice_size))  # Array, in dem die 3D-Koord. für jedes 2D-Pixel gepeichert wird. Dimension 0 = welche Koordinate (x, y, z). Dim 1,2 = Pixelpos.
#     coords[0] = center[0] + grid_x * v1[0] + grid_y * v2[0]
#     coords[1] = center[1] + grid_x * v1[1] + grid_y * v2[1]
#     coords[2] = center[2] + grid_x * v1[2] + grid_y * v2[2]

#     coords_offset = np.zeros((3, slice_size, slice_size))  # Array, in dem die 3D-Koord. für jedes 2D-Pixel gepeichert wird. Dimension 0 = welche Koordinate (x, y, z). Dim 1,2 = Pixelpos.
#     coords_offset[0] = center_offset[0] + grid_x * v1[0] + grid_y * v2[0]
#     coords_offset[1] = center_offset[1] + grid_x * v1[1] + grid_y * v2[1]
#     coords_offset[2] = center_offset[2] + grid_x * v1[2] + grid_y * v2[2]

#     return coords, coords_offset

# def random_plane_slice(volume_img, volume_seg, center:np.array, center_offset:np.array, grid_x, grid_y):
#     """Randomly slices the volume image and semgementation in the same way. Outputs different planes than just x,y,z.
#     Outputs: 
#     slice_img, slice_seg: One Datapoint
#     slice_img_offset, slice_seg_offset: Datapoint of the same new random volume but with a offset 
#     """
#     coords, coords_offset = __random_plane_coords(volume_img.shape, center, center_offset, grid_x, grid_y)

#     # Image interpolation 
#     slice_img = map_coordinates(volume_img, coords, order=1, mode='constant')
#     slice_img_offset = map_coordinates(volume_img, coords_offset, order=1, mode='constant')

#     # Segmentation interpolation
#     slice_seg = map_coordinates(volume_seg, coords, order=0, mode='constant')
#     slice_seg_offset = map_coordinates(volume_seg, coords_offset, order=0, mode='constant')
    
#     def unified_crop(img1, seg1, img2, seg2):
#         # Create a combined mask from all four arrays
#         combined_mask = (img1 > 0) | (seg1 > 0) | (img2 > 0) | (seg2 > 0)

#         if not np.any(combined_mask):
#             # Return center portion if no content
#             center_y, center_x = img1.shape[0] // 2, img1.shape[1] // 2
#             crop_size = min(img1.shape) // 3
#             ymin = max(0, center_y - crop_size // 2)
#             ymax = min(img1.shape[0], center_y + crop_size // 2)
#             xmin = max(0, center_x - crop_size // 2)
#             xmax = min(img1.shape[1], center_x + crop_size // 2)
#             return ymin, ymax, xmin, xmax

#         # Find bounding box of combined content
#         rows = np.any(combined_mask, axis=1)
#         cols = np.any(combined_mask, axis=0)

#         ymin, ymax = np.where(rows)[0][[0, -1]]
#         xmin, xmax = np.where(cols)[0][[0, -1]]

#         ymin = max(0, ymin)
#         ymax = min(img1.shape[0], ymax)
#         xmin = max(0, xmin)
#         xmax = min(img1.shape[1], xmax)

#         return ymin, ymax, xmin, xmax

#     # Get unified cropping coordinates
#     ymin, ymax, xmin, xmax = unified_crop(slice_img, slice_seg, slice_img_offset, slice_seg_offset)

#     # Apply same crop to all slices
#     slice_img = slice_img[ymin:ymax, xmin:xmax]
#     slice_seg = slice_seg[ymin:ymax, xmin:xmax]
#     slice_img_offset = slice_img_offset[ymin:ymax, xmin:xmax]
#     slice_seg_offset = slice_seg_offset[ymin:ymax, xmin:xmax]
    
#     slice_img = tf.convert_to_tensor(slice_img)
#     slice_seg = tf.convert_to_tensor(slice_seg)
#     slice_img_offset = tf.convert_to_tensor(slice_img_offset)
#     slice_seg_offset = tf.convert_to_tensor(slice_seg_offset)

#     return slice_img, slice_seg, slice_img_offset, slice_seg_offset