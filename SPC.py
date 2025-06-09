import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def gen_mask(N, radius_m, size, aper_ext_m):
    x = np.linspace(-aper_ext_m / 2, aper_ext_m / 2, size)
    y = np.linspace(-aper_ext_m / 2, aper_ext_m / 2, size)
    Y, X = np.meshgrid(y, x)

    angle = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)
    theta = 2 * np.pi / N

    mask = np.ones_like(r, dtype=bool)
    for k in range(N):
        angle_k = -np.pi + k * theta
        edge_angle = angle - angle_k
        mask &= (np.cos(edge_angle)) * r <= radius_m

    return mask.astype(float)

st.title("N-gon Diffraction Pattern Simulator")

N = st.slider("N-gon", 3, 20, 6)
radius_mm = st.slider("Radius (mm)", 0.001, 0.1, 0.01, step=0.001)
lamda_nm = st.slider("Wavelength λ (nm)", 400, 700, 632, step=10)
z_m = st.slider("Distance z (m)", 0.1, 2.0, 1.0, step=0.1)

lamda = lamda_nm * 1e-9
radius = radius_mm * 1e-3
aper_size = 1024
aper_ext = radius * 64

aperture = gen_mask(N, radius, aper_size, aper_ext)

fft = np.fft.fftshift(np.fft.fft2(aperture))
intensity = np.abs(fft)**2
intensity /= np.max(intensity)
phase = np.angle(fft)

dx = aper_ext / aper_size
fx = np.fft.fftshift(np.fft.fftfreq(aper_size, d=dx))
x_m = fx * lamda * z_m

mid = intensity.shape[0] // 2
profile = intensity[mid, :]
profile /= np.max(profile)

peaks, _ = find_peaks(profile, height=0.01)
if len(peaks) >= 1:
    peak_spacing_px = np.mean(np.diff(peaks))
    peak_spacing_m = (x_m[1] - x_m[0]) * peak_spacing_px
else:
    peak_spacing_px = peak_spacing_m = float("nan")

st.write(f"**Aperture radius:** {radius_mm:.3f} mm")
st.write(f"**λ = {lamda_nm} nm**, **z = {z_m:.2f} m**")
st.write(f"**Avg. peak spacing:** {peak_spacing_px:.2f} px ≈ {peak_spacing_m:.6f} m")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].imshow(aperture, cmap='gray', extent=[-aper_ext*1e3/2, aper_ext*1e3/2]*2)
axs[0, 0].set_title("Aperture")
axs[0, 0].set_xlabel("x (mm)")
axs[0, 0].set_ylabel("y (mm)")

im = axs[0, 1].imshow(np.log1p(intensity), cmap='inferno',
                extent=[x_m[0]*1e3, x_m[-1]*1e3, x_m[0]*1e3, x_m[-1]*1e3])
im.set_clim((0, 0.01))
axs[0, 1].set_title("Diffraction Pattern (log scale)")
axs[0, 1].set_xlabel("x (mm)")
axs[0, 1].set_ylabel("y (mm)")

axs[1, 0].plot(x_m * 1e3, profile)
axs[1, 0].set_title("Intensity Profile (Center Line)")
axs[1, 0].set_xlabel("Position x (mm)")
axs[1, 0].set_ylabel("Normalized Intensity")
axs[1, 0].grid(True)

axs[1, 1].imshow(phase, cmap='twilight', extent=[-1, 1, -1, 1])
axs[1, 1].set_title("Phase Pattern")

plt.tight_layout()
st.pyplot(fig)
