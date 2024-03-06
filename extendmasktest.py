import numpy as np
import torch as th
from scipy.ndimage import distance_transform_cdt

def extend_mask(binary_mask_, buffer_size):
    """
    Extend a binary mask by setting white pixels that are within a Manhattan distance of 'buffer_size' from a black pixel to black.

    Parameters:
        - binary_mask_ (torch.Tensor): A binary mask tensor of shape (N, 3, H, W), where H is the height and W is the width of the image.
        - buffer_size (int): The Manhattan distance threshold. White pixels within this distance from a black pixel will be set to black.
    """
    binary_mask = binary_mask_.cpu().numpy()

    # Initialize the extended mask array
    extended_masks = []

    # Loop through each image in the binary_mask array
    for image_index in range(binary_mask.shape[0]):
        # Compute the distance transform of the black pixels in the current image
        dist_transform = distance_transform_cdt(binary_mask[image_index, 0, :, :])  # Ignore channels
        print(dist_transform)
        # Find white pixels within a Manhattan distance of 4 from a black pixel
        extended_mask = np.ones_like(binary_mask[image_index, 0, :, :]) * binary_mask[image_index, 0, :, :]

        # Get the dimensions of the current image
        H, W = extended_mask.shape

        # Iterate over each pixel in the current image
        for i in range(H):
            for j in range(W):
                if binary_mask[image_index, 0, i, j] == 1 and dist_transform[i,j] <= buffer_size:  # White pixel but close enough to mask
                    extended_mask[i, j] = 0
        
        # Append the extended mask for the current image to the list
        extended_masks.append(extended_mask)

    # Convert the list of extended masks into a numpy array along the first axis
    extended_masks_array = np.array(extended_masks)

    # Repeat the extended mask along the channel axis to make it compatible with RGB format
    extended_masks_rgb = np.repeat(extended_masks_array[:, np.newaxis, :, :], 3, axis=1)
    
    return th.tensor(extended_masks_rgb, requires_grad=False)


def heated_mask(binary_mask_, device):
    """
    Create heated masks that is 1 at the center and a decimal value the farther away

    Parameters:
        - binary_mask_ (torch.Tensor): A binary mask tensor of shape (N, 3, H, W), where H is the height and W is the width of the image.
    """
    binary_mask = binary_mask_.cpu().numpy()

    # Initialize the extended mask array
    heated_masks = []

    # Loop through each image in the binary_mask array
    for image_index in range(binary_mask.shape[0]):
        # Compute the distance transform of the black pixels in the current image
        dist_transform = distance_transform_cdt(1 - binary_mask[image_index, 0, :, :])  # Ignore channels
        
        heated_masks.append(dist_transform) #/ np.max(dist_transform))

    # Convert the list of heated masks into a numpy array along the first axis
    heated_masks_array = np.array(heated_masks)

    # Repeat the heated mask along the channel axis to make it compatible with RGB format
    heated_masks_rgb = np.repeat(heated_masks_array[:, np.newaxis, :, :], 3, axis=1)
    
    return th.tensor(heated_masks_rgb, requires_grad=False)

  

def main():
    binary_mask = th.tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    # binary_mask = th.tensor([
    # [1, 1, 1, 1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 0, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1, 1, 1, 1],
    # [1, 1, 1, 1, 1, 1, 1, 1, 1]])


    binary_mask_expanded = binary_mask.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)

    # Extend the mask
    #extended_mask = extend_mask(binary_mask_expanded, 3)

    extended_mask = heated_mask(binary_mask_expanded, 3)

    

    # Print the original and extended masks
    print("Original Binary Mask:")
    print(binary_mask)

    print("\nExtended Mask:")
    print(extended_mask[0,0,:,:])

if __name__ == "__main__":
    main()
