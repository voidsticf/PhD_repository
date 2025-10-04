import numpy as np
import cv2
import imageio
from scipy.fftpack import fft2, ifft2, fftshift
import scipy.ndimage as ndi
from PIL import Image
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
import os 




def image_division_function(path_of_the_images_to_process,path_to_copy_all_changed_images):
    
    ###################################################################################################
    #### Gaussian window in order to make a smoother transition to the edges of the image ####
    def apply_gaussian_window(image, sigma=0.4):
        M, N = image.shape[:2]  # Get image size
        
        # Generate 1D Gaussian distributions
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, M)
        X, Y = np.meshgrid(x, y)
        
        # Create 2D Gaussian window
        gaussian_window = np.exp(-((X**2 + Y**2) / (2 * sigma**2)))
        
        # Apply the window
        image_windowed = image.astype(np.float32) * gaussian_window
        
        # Normalize back to uint8
        image_windowed = cv2.normalize(image_windowed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return image_windowed
    ###################################################################################################    

    ################################################################################################### 
    ### Facilitate the shift - it shifts an image by dx pixels horizontally and dy pixels vertically ###    
    def shift_image(image, dx, dy):


    #  Parameters:
    #    image (numpy.ndarray): The input image array.
    #    dx (int): The number of pixels to shift horizontally.
    #    dy (int): The number of pixels to shift vertically.
           
        
        shifted_image = ndi.shift(image, shift=(dy, dx), mode='nearest')  # mode='nearest' to handle edges
        return shifted_image
    
    
        ####### MAIN FUNCTION #########
    def phase_correlation(image1, image2):
    # Convert images to grayscale (if not already)
    # Compute Fourier Transforms

        F1 = np.fft.fft2(image1)
        F2 = np.fft.fft2(image2)
        #### Compute Cross-Power Spectrum ####
        R = (F1 * np.conj(F2)) / np.abs(F1 * np.conj(F2))
        #### Compute inverse FFT to obtain phase correlation ####
        correlation = np.fft.ifft2(R).real

        #### Find peak location (shift) ###
        dy, dx = np.unravel_index(np.argmax((correlation)), correlation.shape)
        ### Adjust for wrap-around effect ###
        if dy > image1.shape[0] // 2:
            dy -= image1.shape[0]
        if dx > image1.shape[1] // 2:
            dx -= image1.shape[1]

        return dx, dy, correlation
    ###################################################################################################    

    ###################################################################################################    
    ### This section could be modified according to your need ###
    list_of_images=os.listdir(path_of_the_images_to_process) 
    filtered_images=[f for f in list_of_images if f.startswith('...')]
    sorted_images=sorted(filtered_images, key=lambda x: int(re.search(r'...',x).group(1)))
    shot_images=[f for f in sorted_images if '... in f]
    reference_images=[f for f in sorted_images if '....' not in f]
    ###################################################################################################    

    ###################################################################################################    
    for i in range(len(shot_images)):
        os.chdir(path_of_the_images_to_process) ###could be omitted, depending on your needs
        reference = cv2.imread(reference_images[i], cv2.IMREAD_GRAYSCALE)
        shotimage = cv2.imread(shot_images[i], cv2.IMREAD_GRAYSCALE)
        dx,dy, g=phase_correlation(apply_gaussian_window(reference[0:1600,:]), apply_gaussian_window(shotimage[0:1600,:]))
        shifted_reference_image=shift_image(reference, -dx, -dy)#minus for the reference shot, plus for the shot image
    ###################################################################################################    
            
    ###################################################################################################    
        ### Divide the images in order to remove the background ###
        # Copy of the divided image to a specific file
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for division by zero
            result = np.divide(shotimage,shifted_reference_image )
            result[~np.isfinite(result)] = 0
        # Convert the result to a PIL Image
        result_image = Image.fromarray(result)
        
        # Save the result as a TIFF file
        imageio.imwrite(reference_images[i][:-4]+'_shifted.tif',shifted_reference_image) 
        string_to_save=shot_images[i][:-4]+'_divided.tif'
        result_image.save(string_to_save)
        
    ###################################################################################################    
    
