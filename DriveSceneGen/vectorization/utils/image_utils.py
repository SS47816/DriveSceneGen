import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def combine_dx_dy(dx: float, dy: float, mx: float, my: float, thresh: float=0.1) -> int:
    if np.fabs(dx - mx) <= thresh and np.fabs(dy - my) <= thresh:
        return 0
    else:
        return 255
    

def get_gray_image(img_color: Image, plot: bool=False) -> Image:
    img_tensor = np.array(img_color, dtype=float)

    x_channel = img_tensor[:, :, 0] / 255.0
    y_channel = img_tensor[:, :, 1] / 255.0
    z_channel = img_tensor[:, :, 2] / 255.0

    ## flatten the image
    dx_gray_values = x_channel.flatten()
    dy_gray_values = y_channel.flatten()
    dz_gray_values = z_channel.flatten()

    ## compute the histogram
    dx_histogram, dxbins = np.histogram(dx_gray_values, bins=256, range=(0, 1))
    dy_histogram, dybins = np.histogram(dy_gray_values, bins=256, range=(0, 1))
    dz_histogram, dzbins = np.histogram(dz_gray_values, bins=256, range=(0, 1))

    ## find the maximum peak of the histogram
    dx_max_gray_value = dxbins[np.argmax(dx_histogram)]
    dy_max_gray_value = dybins[np.argmax(dy_histogram)]
    dz_max_gray_value = dybins[np.argmax(dz_histogram)]

    ## compute the mean of the maximum peak
    mx = dx_max_gray_value
    my = dy_max_gray_value
    mz = dz_max_gray_value

    img_mask = [combine_dx_dy(dx, dy, mx, my) for (dx, dy) in zip(dx_gray_values, dy_gray_values)]
    img_mask_tensor = np.array(img_mask, dtype=np.uint8).reshape(img_tensor.shape[:2])
    img_gray_tensor = np.stack((img_mask_tensor, img_mask_tensor, img_mask_tensor), axis=-1)
    img_gray = Image.fromarray(img_gray_tensor)

    ## plot the histogram
    if plot:
        fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(8, 8), sharex='row', sharey='row')
        ax0.bar(dxbins[:-1], dx_histogram, width=1/256, align='edge', color='r')
        ax0.set_title('x_channel')

        ax1.bar(dybins[:-1], dy_histogram, width=1/256, align='edge', color='g')
        ax1.set_title('y_channel')

        ax2.bar(dzbins[:-1], dz_histogram, width=1/256, align='edge', color='b')
        ax2.set_title('z_channel')

        ax3.imshow(x_channel, cmap='Reds')
        ax4.imshow(y_channel, cmap='Greens')
        ax5.imshow(z_channel, cmap='Blues')

        plt.tight_layout()
        plt.show()

    return img_gray