
def snr(img, smask, nmask=None):
    fg_mean = img[smask > 0].mean()

    if nmask is None:
        nmask = smask

    bg_std = img[nmask > 0].std()
    return float(fg_mean / bg_std)