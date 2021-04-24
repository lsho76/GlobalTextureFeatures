This is the Python code for the paper: 

Lee SH, Kao GD, Feigenberg SJ, et al. Multi-block discriminant analysis of integrative 18F-FDG-PET/CT radiomics for predicting circulating tumor cells in early stage non-small cell lung cancer treated with stereotactic body radiation therapy. International Journal of Radiation Oncology, Biology, Physics. 2021. doi: https://doi.org/10.1016/j.ijrobp.2021.02.030.

If you input 3D image and mask in simpleITK format to the extractMomentInvariantFeatures function, 3D global texture features can be extracted after image intensity values are rescaled to 0-255, which consist of the 2nd, 3rd and 4th-order normalized central moments and moment invariants. See the following paper for details about their computational formulas.

Lee SH, Kim JH, Cho N, et al. Multilevel analysis of spatiotemporal association features for differentiation of tumor enhancement patterns in breast DCE-MRI. Medical Physics. 2010;37(8):3940-56. doi: https://doi.org/10.1118/1.3446799.
