from __future__ import absolute_import, division, print_function, unicode_literals
from math import pi
from six import string_types
import numpy as np
import scipy.ndimage as nd

FSL_FAST_LABELS = {'csf': 1, 'gm': 2, 'wm': 3, 'bg': 0}

def snr(img, smask, nmask=None, erode=True, fglabel=1):
    """
    Calculate SNR

    >>> expected_snr = 0.5
    >>> img = np.zeros((50, 50, 50))
    >>> img[10:40, 10:40, 10:40] = 1.0
    >>> smask = img.copy()
    >>> img *= 100
    >>> np.random.seed(1234)
    >>> img += np.random.normal(0.0, 1./expected_snr, size=(50, 50, 50)) * 100 * smask
    >>> calculated_snr = snr(img, smask, erode=False)
    >>> abs(calculated_snr - expected_snr) < 0.01
    True


    """
    fgmask = _prepare_mask(smask, fglabel, erode)
    bgmask = _prepare_mask(nmask, 1, erode) if nmask is not None else None

    fg_mean = np.median(img[fgmask > 0])
    if bgmask is None:
        bgmask = fgmask
        bg_mean = fg_mean
        # Manually compute sigma, using Bessel's correction (the - 1 in the normalizer)
        bg_std = np.sqrt(np.sum((img[bgmask > 0] - bg_mean) ** 2) / (np.sum(bgmask) - 1))
    else:
        bg_std = np.sqrt(2.0/(4.0 - pi)) * img[bgmask > 0].std(ddof=1)

    return float(fg_mean / bg_std)

def _prepare_mask(mask, label, erode=True):
    fgmask = mask.copy()

    if np.issubdtype(fgmask.dtype, np.integer):
        if isinstance(label, string_types):
            label = FSL_FAST_LABELS[label]

        fgmask[fgmask != label] = 0
        fgmask[fgmask == label] = 1
    else:
        fgmask[fgmask > .95] = 1.
        fgmask[fgmask < 1.] = 0

    if erode:
        # Create a structural element to be used in an opening operation.
        struc = nd.generate_binary_structure(3, 2)
        # Perform an opening operation on the background data.
        fgmask = nd.binary_opening(fgmask, structure=struc).astype(np.uint8)

    return fgmask
