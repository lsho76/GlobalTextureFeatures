import os
import SimpleITK as sitk
import numpy as np
from numpy import mgrid, sum
import math

# image, mask: images in simple ITK format

def moments3d(image, mask):
    
    # Change from sitk image to numpy array
    image_array = sitk.GetArrayViewFromImage(image) 
    mask_array = sitk.GetArrayViewFromImage(mask)

    # Extract image textures within mask volume 
    roiImage_array = image_array * mask_array

    assert len(roiImage_array.shape) == 3
    x, y, z = mgrid[:roiImage_array.shape[0],:roiImage_array.shape[1],:roiImage_array.shape[2]] # Get a dense multi-dimensional 'meshgrid'.

    # Get image spacing in x, y and z
    dx = image.GetSpacing()[0]
    dy = image.GetSpacing()[1]
    dz = image.GetSpacing()[2]

    Ui1 = x*dx
    Ui0 = (x-1)*dx

    Vi1 = y*dy
    Vi0 = (y-1)*dy

    Wi1 = z*dz
    Wi0 = (z-1)*dz

    Ixi1 = 0.5*(Ui1**2 - Ui0**2)
    Ixi0 = Ui1 - Ui0

    Iyi1 = 0.5*(Vi1**2 - Vi0**2)
    Iyi0 = Vi1 - Vi0

    Izi1 = 0.5*(Wi1**2 - Wi0**2)
    Izi0 = Wi1 - Wi0

    m100 = sum(Ixi1*Iyi0*Izi0*roiImage_array)
    m010 = sum(Ixi0*Iyi1*Izi0*roiImage_array)
    m001 = sum(Ixi0*Iyi0*Izi1*roiImage_array)
    m000 = sum(Ixi0*Iyi0*Izi0*roiImage_array)

    moments = {}
    moments['mean_x'] = m100/m000
    moments['mean_y'] = m010/m000
    moments['mean_z'] = m001/m000

    U1 = x*dx - moments['mean_x']
    U0 = (x-1)*dx - moments['mean_x']

    V1 = y*dy - moments['mean_y']
    V0 = (y-1)*dy - moments['mean_y']

    W1 = z*dz - moments['mean_z']
    W0 = (z-1)*dz - moments['mean_z']

    Ix0 = U1 - U0
    Iy0 = V1 - V0
    Iz0 = W1 - W0

    Ix1 = 1.0/2.0*(U1**2 - U0**2)
    Iy1 = 1.0/2.0*(V1**2 - V0**2)
    Iz1 = 1.0/2.0*(W1**2 - W0**2)

    Ix2 = 1.0/3.0*(U1**3 - U0**3)
    Iy2 = 1.0/3.0*(V1**3 - V0**3)
    Iz2 = 1.0/3.0*(W1**3 - W0**3)

    Ix3 = 1.0/4.0*(U1**4 - U0**4)
    Iy3 = 1.0/4.0*(V1**4 - V0**4)
    Iz3 = 1.0/4.0*(W1**4 - W0**4)

    Ix4 = 1.0/5.0*(U1**5 - U0**5)
    Iy4 = 1.0/5.0*(V1**5 - V0**5)
    Iz4 = 1.0/5.0*(W1**5 - W0**5)

    mu000 = sum(Ix0*Iy0*Iz0*roiImage_array)
    mu200 = sum(Ix2*Iy0*Iz0*roiImage_array)
    mu020 = sum(Ix0*Iy2*Iz0*roiImage_array)
    mu002 = sum(Ix0*Iy0*Iz2*roiImage_array)

    mu300 = sum(Ix3*Iy0*Iz0*roiImage_array)
    mu030 = sum(Ix0*Iy3*Iz0*roiImage_array)
    mu003 = sum(Ix0*Iy0*Iz3*roiImage_array)

    mu400 = sum(Ix4*Iy0*Iz0*roiImage_array)
    mu040 = sum(Ix0*Iy4*Iz0*roiImage_array)
    mu004 = sum(Ix0*Iy0*Iz4*roiImage_array)

    mu101 = sum(Ix1*Iy0*Iz1*roiImage_array)
    mu110 = sum(Ix1*Iy1*Iz0*roiImage_array)
    mu011 = sum(Ix0*Iy1*Iz1*roiImage_array)

    mu210 = sum(Ix2*Iy1*Iz0*roiImage_array)
    mu201 = sum(Ix2*Iy0*Iz1*roiImage_array)
    mu120 = sum(Ix1*Iy2*Iz0*roiImage_array)

    mu111 = sum(Ix1*Iy1*Iz1*roiImage_array)
    mu102 = sum(Ix1*Iy0*Iz2*roiImage_array)
    mu021 = sum(Ix0*Iy2*Iz1*roiImage_array)
    mu012 = sum(Ix0*Iy1*Iz2*roiImage_array)

    mu310 = sum(Ix3*Iy1*Iz0*roiImage_array)
    mu301 = sum(Ix3*Iy0*Iz1*roiImage_array)
    mu220 = sum(Ix2*Iy2*Iz0*roiImage_array)
    mu211 = sum(Ix2*Iy1*Iz1*roiImage_array)
    mu202 = sum(Ix2*Iy0*Iz2*roiImage_array)
    mu130 = sum(Ix1*Iy3*Iz0*roiImage_array)
    mu121 = sum(Ix1*Iy2*Iz1*roiImage_array)
    mu112 = sum(Ix1*Iy1*Iz2*roiImage_array)
    mu103 = sum(Ix1*Iy0*Iz3*roiImage_array)
    mu031 = sum(Ix0*Iy3*Iz1*roiImage_array)
    mu022 = sum(Ix0*Iy2*Iz2*roiImage_array)
    mu013 = sum(Ix0*Iy1*Iz3*roiImage_array)

    r2 = 2.0/3.0+1.0
    r3 = 3.0/3.0+1.0
    r4 = 4.0/3.0+1.0

    nu200 = mu200/math.pow(mu000,r2)
    nu020 = mu020/math.pow(mu000,r2)
    nu002 = mu002/math.pow(mu000,r2)

    nu300 = mu300/math.pow(mu000,r3)
    nu030 = mu030/math.pow(mu000,r3)
    nu003 = mu003/math.pow(mu000,r3)

    nu400 = mu400/math.pow(mu000,r4)
    nu040 = mu040/math.pow(mu000,r4)
    nu004 = mu004/math.pow(mu000,r4)

    nu101 = mu101/math.pow(mu000,r2)
    nu110 = mu110/math.pow(mu000,r2)
    nu011 = mu011/math.pow(mu000,r2)

    nu210 = mu210/math.pow(mu000,r3)
    nu201 = mu201/math.pow(mu000,r3)
    nu120 = mu120/math.pow(mu000,r3)

    nu111 = mu111/math.pow(mu000,r3)
    nu102 = mu102/math.pow(mu000,r3)
    nu021 = mu021/math.pow(mu000,r3)
    nu012 = mu012/math.pow(mu000,r3)

    nu310 = mu310/math.pow(mu000,r4)
    nu301 = mu301/math.pow(mu000,r4)
    nu220 = mu220/math.pow(mu000,r4)
    nu211 = mu211/math.pow(mu000,r4)

    nu202 = mu202/math.pow(mu000,r4)
    nu130 = mu130/math.pow(mu000,r4)
    nu121 = mu121/math.pow(mu000,r4)
    nu112 = mu112/math.pow(mu000,r4)
    nu103 = mu103/math.pow(mu000,r4)
    nu031 = mu031/math.pow(mu000,r4)
    nu022 = mu022/math.pow(mu000,r4)
    nu013 = mu013/math.pow(mu000,r4)

    M21 = nu200 + nu020 + nu002
    M22 = nu200*nu020 + nu200*nu002 + nu020*nu002 - math.pow(nu101,2) - math.pow(nu110,2) - math.pow(nu011,2)
    M23 = nu200*nu020*nu002 - nu002*math.pow(nu110,2) + 2*nu110*nu101*nu011 - nu020*math.pow(nu101,2) - nu200*math.pow(nu011,2)
    M3 = math.pow(nu300,2) + math.pow(nu030,2) + math.pow(nu003,2) + 3*math.pow(nu210,2) + 3*math.pow(nu201,2) + 3*math.pow(nu120,2) + 6*math.pow(nu111,2) + 3*math.pow(nu102,2) + 3*math.pow(nu021,2) + 3*math.pow(nu012,2)
    M4 = math.pow(nu400,2) + math.pow(nu040,2) + math.pow(nu004,2) + 4*math.pow(nu310,2) + 4*math.pow(nu301,2) + 6*math.pow(nu220,2) + 12*math.pow(nu211,2) + 6*math.pow(nu202,2) + 4*math.pow(nu130,2) + 12*math.pow(nu121,2) + 12*math.pow(nu112,2) + 4*math.pow(nu103,2) + 4*math.pow(nu031,2) + 6*math.pow(nu022,2) + 4*math.pow(nu013,2)

    return nu200, nu020, nu002, nu101, nu110, nu011, nu300, nu030, nu003, nu210, nu201, nu120, nu111, nu102, nu021, nu012, nu400, nu040, nu004, nu310, nu301, nu220, nu211, nu202, nu130, nu121, nu112, nu103, nu031, nu022, nu013, M21, M22, M23, M3, M4

def extractMomentInvariantFeatures(image, mask, RTstructure, OncoID, filepath):
    import csv
    os.chdir(filepath)

    with open(os.path.join(str(OncoID)+'_'+RTstructure+'_momentinvariants.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        decompositionName = 'original'
        rescaler1 = sitk.RescaleIntensityImageFilter()
        rescaler1.SetOutputMinimum(0)
        rescaler1.SetOutputMaximum(255)
        rescaled_image = rescaler1.Execute(image)
        headers_original = list([decompositionName+'_eta200', decompositionName+'_eta020', decompositionName+'_eta002', decompositionName+'_eta101', decompositionName+'_eta110',
                            decompositionName+'_eta011', decompositionName+'_eta300', decompositionName+'_eta030', decompositionName+'_eta003', decompositionName+'_eta210',
                            decompositionName+'_eta201', decompositionName+'_eta120', decompositionName+'_eta111', decompositionName+'_eta102', decompositionName+'_eta021',
                            decompositionName+'_eta012', decompositionName+'_eta400', decompositionName+'_eta040', decompositionName+'_eta004', decompositionName+'_eta310',
                            decompositionName+'_eta301', decompositionName+'_eta220', decompositionName+'_eta211', decompositionName+'_eta202', decompositionName+'_eta130',
                            decompositionName+'_eta121', decompositionName+'_eta112', decompositionName+'_eta103', decompositionName+'_eta031', decompositionName+'_eta022', decompositionName+'_eta013',
                            decompositionName+'_M21', decompositionName+'_M22', decompositionName+'_M23', decompositionName+'_M3', decompositionName+'_M4'])

        print('Calculated moment invariant features with', decompositionName)
        nu200, nu020, nu002, nu101, nu110, nu011, nu300, nu030, nu003, nu210, nu201, nu120, nu111, nu102, nu021, nu012, nu400, nu040, nu004, nu310, nu301, nu220, nu211, nu202, nu130, nu121, nu112, nu103, nu031, nu022, nu013, M21, M22, M23, M3, M4 = moments3d(rescaled_image, mask)
        moments = np.array([nu200, nu020, nu002, nu101, nu110, nu011, nu300, nu030, nu003, nu210, nu201, nu120, nu111, nu102, nu021, nu012, nu400, nu040, nu004, nu310, nu301, nu220, nu211, nu202, nu130, nu121, nu112, nu103, nu031, nu022, nu013, M21, M22, M23, M3, M4])

        headers_wavelet = []
        for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image, mask):
            print('Calculated moment invariant features with', decompositionName)
            rescaler2 = sitk.RescaleIntensityImageFilter()
            rescaler2.SetOutputMinimum(0)
            rescaler2.SetOutputMaximum(255)
            rescaled_decompositionImage = rescaler2.Execute(decompositionImage)
            nu200, nu020, nu002, nu101, nu110, nu011, nu300, nu030, nu003, nu210, nu201, nu120, nu111, nu102, nu021, nu012, nu400, nu040, nu004, nu310, nu301, nu220, nu211, nu202, nu130, nu121, nu112, nu103, nu031, nu022, nu013, M21, M22, M23, M3, M4 = moments3d(rescaled_decompositionImage, mask)

            headers_wavelet += [decompositionName+'_eta200', decompositionName+'_eta020', decompositionName+'_eta002', decompositionName+'_eta101', decompositionName+'_eta110',
                        decompositionName+'_eta011', decompositionName+'_eta300', decompositionName+'_eta030', decompositionName+'_eta003', decompositionName+'_eta210',
                        decompositionName+'_eta201', decompositionName+'_eta120', decompositionName+'_eta111', decompositionName+'_eta102', decompositionName+'_eta021',
                        decompositionName+'_eta012', decompositionName+'_eta400', decompositionName+'_eta040', decompositionName+'_eta004', decompositionName+'_eta310',
                        decompositionName+'_eta301', decompositionName+'_eta220', decompositionName+'_eta211', decompositionName+'_eta202', decompositionName+'_eta130',
                        decompositionName+'_eta121', decompositionName+'_eta112', decompositionName+'_eta103', decompositionName+'_eta031', decompositionName+'_eta022', decompositionName+'_eta013',
                        decompositionName+'_M21', decompositionName+'_M22', decompositionName+'_M23', decompositionName+'_M3', decompositionName+'_M4']

            moments = np.concatenate([moments,np.array([nu200, nu020, nu002, nu101, nu110, nu011, nu300, nu030, nu003, nu210, nu201, nu120, nu111, nu102, nu021, nu012, nu400, nu040, nu004, nu310, nu301, nu220, nu211, nu202, nu130, nu121, nu112, nu103, nu031, nu022, nu013, M21, M22, M23, M3, M4])])

        logFeatures = {}
        sigmaValues = [1,3,5]
        headers_LoG = []
        for logImage, decompositionName, inputSettings in imageoperations.getLoGImage(image, mask, sigma=sigmaValues):

            print('Calculated moment invariant features with', decompositionName)
            rescaler3 = sitk.RescaleIntensityImageFilter()
            rescaler3.SetOutputMinimum(0)
            rescaler3.SetOutputMaximum(255)
            rescaled_logImage = rescaler3.Execute(logImage)
            nu200, nu020, nu002, nu101, nu110, nu011, nu300, nu030, nu003, nu210, nu201, nu120, nu111, nu102, nu021, nu012, nu400, nu040, nu004, nu310, nu301, nu220, nu211, nu202, nu130, nu121, nu112, nu103, nu031, nu022, nu013, M21, M22, M23, M3, M4 = moments3d(rescaled_logImage, mask)

            headers_LoG += [decompositionName+'_eta200', decompositionName+'_eta020', decompositionName+'_eta002', decompositionName+'_eta101', decompositionName+'_eta110',
                            decompositionName+'_eta011', decompositionName+'_eta300', decompositionName+'_eta030', decompositionName+'_eta003', decompositionName+'_eta210',
                            decompositionName+'_eta201', decompositionName+'_eta120', decompositionName+'_eta111', decompositionName+'_eta102', decompositionName+'_eta021',
                            decompositionName+'_eta012', decompositionName+'_eta400', decompositionName+'_eta040', decompositionName+'_eta004', decompositionName+'_eta310',
                            decompositionName+'_eta301', decompositionName+'_eta220', decompositionName+'_eta211', decompositionName+'_eta202', decompositionName+'_eta130',
                            decompositionName+'_eta121', decompositionName+'_eta112', decompositionName+'_eta103', decompositionName+'_eta031', decompositionName+'_eta022', decompositionName+'_eta013',
                            decompositionName+'_M21', decompositionName+'_M22', decompositionName+'_M23', decompositionName+'_M3', decompositionName+'_M4']

            moments = np.concatenate([moments,np.array([nu200, nu020, nu002, nu101, nu110, nu011, nu300, nu030, nu003, nu210, nu201, nu120, nu111, nu102, nu021, nu012, nu400, nu040, nu004, nu310, nu301, nu220, nu211, nu202, nu130, nu121, nu112, nu103, nu031, nu022, nu013, M21, M22, M23, M3, M4])])

        headers = headers_original+headers_wavelet+headers_LoG
        wr.writerow(headers)
        row = []
        index = 0
        for h in headers:
            row.append(moments[index])
            index += 1
        wr.writerow(row)
    f.close()
