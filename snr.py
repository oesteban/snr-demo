

def snr(img, smask, nmask):
    fg_mean = img[smask > 0].mean()
    bg_std = img[nmask > 0].std()
    return float(fg_mean / bg_std)
