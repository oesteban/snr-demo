
def snr(img):
    fg_mean = img.mean()
    bg_std = img.std()
    return float(fg_mean / bg_std)
