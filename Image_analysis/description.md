### Information for the script

#### Goal:
The main use of the code is  the reference and the shot images show an offset. In that sense, this codes corrects this **offset** and **divides the images**. The steps that you need to follow in order to run this code is:
- Define the path of the images to process (*path_of_the_images_to_process*)
- Define the path to save the processed images (*path_to_copy_all_changed_images*)

##### Code Comment:

It finds the phase correlation of the shot image with respect to the reference one. Consequently,
- $\Delta x=x_{ref}-x_{shot}$
- $\Delta y=y_{ref}-y_{shot}$


We need to be careful and correct for the $\textbf{Wrap-Around Effect}$. Τhe phase correlation process involves computing the cross-power spectrum in the Fourier domain and performing an inverse FFT to obtain a correlation matrix. The peak in this matrix represents the shift between the two images. The correlation peak can sometimes appear at the edges of the correlation matrix instead of near the center. This happens because the FFT assumes $\textbf{periodic boundary conditions}$, meaning it treats the image as if it wraps around at the edges.

##### Wrap-Around Effect correction code:

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

#### Fill Important Information

Before running the script one should change the below commands according to their needs. It is not important to use all the lines of code. However, they should verify that the lists <em>shot_images</em> and  <em>reference_images</em> are set correctly.

*list_of_images = os.listdir(path_of_the_images_to_process)*

*filtered_images=[f for f in list_of_images if f.startswith('')]*

*sorted_images=sorted(filtered_images, key=lambda x: int(re.search(r'',x).group(1)))*

*shot_images=[f for f in sorted_images if '' in f]*

*reference_images=[f for f in sorted_images if '' not in f]*
