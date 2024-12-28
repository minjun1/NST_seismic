# Realistic Synthetic Seismic Data via Neural Style Transfer

This project applies Neural Style Transfer (NST) to create realistic synthetic seismic data by blending the content of a synthetic image with the style of a real seismic image.

Paper: [https://library.seg.org/doi/abs/10.1190/image2022-3739151.1]

---

# Model weights
Download 'vgg_conv.pth' using below link and put it into the 'Model' directory. 
[https://drive.google.com/file/d/1eXErTqV_4fDeIiDvjGMFofhRZT3FPf4z/view?usp=drive_link]


# This file

1. **`data/realimg.npy`** and **`data/synimg1.npy`**: Example NumPy arrays for real and synthetic seismic images.  
2. **`Models/nst.py`**: Defines the VGG network, GramMatrix, and GramMSELoss used for style transfer.  
3. **`vgg_conv.pth`**: Pre-trained VGG weights (required for NST).  
4. **`helper.py`**: Utility functions for data cropping and KL divergence.  
5. **`nst_functions.py`**: Contains the core NST training function (`run_nst()`).  
6. **`NST_demo.ipynb`**: A Jupyter Notebook demonstrating:
   - Data loading and visualization  
   - NST hyperparameters and optimization  
   - Final stylized images  
   - (Optional) T-SNE + KL divergence comparisons  

---


# References

Jing, Yongcheng, et al. "Neural style transfer: A review." IEEE transactions on visualization and computer graphics 26.11 (2019): 3365-3385.
[https://ieeexplore.ieee.org/abstract/document/8732370?casa_token=2BFcYkmpyhoAAAAA:eYdGlsqCLzqTliL6Z4EupDxns8-jJSyVpaziKIcyMA6Q4KgxW8f-LnE05IgoAqihCIVVsoA]


