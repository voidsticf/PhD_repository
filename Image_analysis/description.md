### Information for the script

#### Goal:
Image analysis between two sets of images (reference and main) that display an offset relative to them.

##### Code Comment:

It finds the phase correlation of the shot image with respect to the reference one. Consequently,
- $\Delta x=x_{ref}-x_{shot}$
- $\Delta y=y_{ref}-y_{shot}$


We need to be careful and correct for the $\textbf{Wrap-Around Effect}$. Τhe phase correlation process involves computing the cross-power spectrum in the Fourier domain and performing an inverse FFT to obtain a correlation matrix. The peak in this matrix represents the shift between the two images. The correlation peak can sometimes appear at the edges of the correlation matrix instead of near the center. This happens because the FFT assumes $\textbf{periodic boundary conditions}$, meaning it treats the image as if it wraps around at the edges.

##### Correction code:

-   correlation = np.fft.ifft2(R).real<br>
    dy, dx = np.unravel_index(np.argmax(correlation), correlation.shape)  
    if dy > image1.shape[0] // 2:<br>
         dy -= image1.shape[0]<br>
    if dx > image1.shape[1] // 2:<br>
        dx -= image1.shape[1]

or

-   correlation = np.fft.fftshift(correlation)<br>
    dy, dx = np.unravel_index(np.argmax(correlation), correlation.shape)  
    dy -= correlation.shape[0] // 2
    dx -= correlation.shape[1] // 2

