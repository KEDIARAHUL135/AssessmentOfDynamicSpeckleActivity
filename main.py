import os
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt


def ShowImage(Image, Title, ShowMap=True, isGray=False):
    plt.title(Title)
    if isGray:
        imgPlot = plt.imshow(Image, cmap='gray')
    else:
        imgPlot = plt.imshow(Image)
    if ShowMap:
        imgPlot.set_cmap('nipy_spectral')
    plt.colorbar()
        

# Parsing the arguments
def ArgParse():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("ImageDirPath", type=str,
                    help="Path of the image directory containing the images.")
    args = vars(ap.parse_args())            # Converting it to dictionary.
    
    return args
    

def ReadImages(ImageDirPath):
    if not os.path.isdir(ImageDirPath):
        print("Provide image directory path only.")
        print(f"Provided path {ImageDirPath} does not exists or is not a directory.")
        print("Exiting Code!!!")
        exit(0)

    ImageNames = os.listdir(ImageDirPath)
    ImageNames.sort(key = lambda x : int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    Images = []
    for ImageName in ImageNames:
        Image = cv2.imread(os.path.join(ImageDirPath, ImageName))
        Images.append(Image[:, :, [2]])

    return Images


def GetTHSPImage(SpeckleImages):
    THSP_Image = np.zeros((SpeckleImages[0].shape[0], len(SpeckleImages), 1), dtype=np.uint8)
    
    ColNo = SpeckleImages[0].shape[1] // 2
    for i, Image in enumerate(SpeckleImages):
        THSP_Image[:, [i]] = Image[:, [ColNo]]

    return THSP_Image


def GetCOM(THSP_Image):
    COM = np.zeros((256, 256), dtype=np.uint8)

    for i in range(THSP_Image.shape[0]):
        for j in range(THSP_Image.shape[1]-1):
            COM[THSP_Image[i][j][0]][THSP_Image[i][j+1][0]] += 1

    return COM


def NormalizeCOM(COM):
    Sum = np.sum(COM)
    Normalized_COM = np.asarray(COM.copy(), dtype=np.float32)

    for i in range(len(Normalized_COM)):
        for j in range(len(Normalized_COM[i])):
            if Normalized_COM[i][j] != 0:
                Normalized_COM[i][j] /= Sum

    return Normalized_COM


def CalculateIM_AVD(COM):
    IM = 0.0
    AVD = 0.0

    for i in range(len(COM)):
        for j in range(len(COM[i])):
            if COM[i][j] != 0:
                IM += COM[i][j] * ((i-j)**2)        # IM
                AVD += COM[i][j] * abs(i-j)         # AVD

    return IM, AVD


def GetGDs(SpeckleImages, p=3, p_al=4):
    # Initializing empty array
    GD = np.zeros(SpeckleImages[0].shape, dtype=np.float32)
    PGD = np.zeros(SpeckleImages[0].shape, dtype=np.float32)
    AGD = np.zeros(SpeckleImages[0].shape, dtype=np.float32)

    for k in range(len(SpeckleImages)):
        for l in range(k+1, len(SpeckleImages)):
            Diff = np.absolute(np.array(SpeckleImages[k], dtype=np.float32) - np.array(SpeckleImages[l], dtype=np.float32))
            GD += Diff
            PGD += np.power(Diff, p)

            with np.errstate(divide='ignore', invalid='ignore'):
                AGD += np.divide(Diff, np.power(np.array(SpeckleImages[k], 
                                                dtype=np.float32) + np.array(SpeckleImages[l], dtype=np.float32), 1/p_al))
                AGD[AGD == np.inf] = 0
                AGD = np.nan_to_num(AGD)

    return GD, PGD, AGD


def GetFujiis(SpeckleImages, n=5, p=3):
    F = np.zeros(SpeckleImages[0].shape, dtype=np.float32)
    F_al = np.zeros(SpeckleImages[0].shape, dtype=np.float32)
    AvgOfAll = np.zeros(SpeckleImages[0].shape, dtype=np.float32)
    PGAF = np.zeros(SpeckleImages[0].shape, dtype=np.float32)

    for n in range(len(SpeckleImages)):
        AvgOfAll += np.array(SpeckleImages[n], dtype=np.float32)
    AvgOfAll = np.divide(AvgOfAll, len(SpeckleImages))


    for k in range(len(SpeckleImages)-1):
        with np.errstate(divide='ignore', invalid='ignore'):
            F += np.divide(np.absolute(np.array(SpeckleImages[k], dtype=np.float32) - np.array(SpeckleImages[k+1], dtype=np.float32)), 
                               np.array(SpeckleImages[k], dtype=np.float32) + np.array(SpeckleImages[k+1], dtype=np.float32))
            F[F == np.inf] = 0.0
            F = np.nan_to_num(F)

            F_al += np.divide(np.absolute(np.array(SpeckleImages[k], dtype=np.float32) - np.array(SpeckleImages[k+1], dtype=np.float32)), 
                              np.power(np.array(SpeckleImages[k], dtype=np.float32) + np.array(SpeckleImages[k+1], dtype=np.float32), 1/n))
            F_al[F_al == np.inf] = 0.0
            F_al = np.nan_to_num(F_al)

        PGAF += np.power(np.absolute(AvgOfAll - SpeckleImages[k]), p)

    return F, F_al, PGAF


def GetSMIndex(SpeckleImages):
    SM_Index = 0.0

    for k in range(len(SpeckleImages)-1):
        for i in range(SpeckleImages[k].shape[0]):
            for j in range(SpeckleImages[k].shape[1]):
                SM_Index += abs(SpeckleImages[k+1][i][j] - SpeckleImages[k][i][j])

        SM_Index /= (SpeckleImages[k].shape[0] * SpeckleImages[k].shape[1])

    SM_Index /= len(SpeckleImages)

    return SM_Index


def CalculateTexturalParams(SpeckleImages):
    GLCM = GetCOM(SpeckleImages[0])
    GLCM = NormalizeCOM(GLCM)

    Contrast, Homogeneity, AngularSecondMoment = 0.0, 0.0, 0.0
    for i in range(GLCM.shape[0]):
        for j in range(GLCM.shape[1]):
            Contrast += (GLCM[i][j] * ((j-i)**2))

            Homogeneity += (GLCM[i][j] / (1 + (i - j)**2))

            AngularSecondMoment += (GLCM[i][j]**2)

    return Contrast, Homogeneity, AngularSecondMoment


def ProcessSpecklePattern(SpeckleImages):
    # Getting THSP image from the speckle images
    THSP_Image = GetTHSPImage(SpeckleImages)

    # Finding the COM
    COM = GetCOM(THSP_Image)

    # Normalizing COM for IM calculation
    Normalized_COM = NormalizeCOM(COM)
    
    # Displaying THSP Image, and COM
    plt.subplot(121)
    ShowImage(THSP_Image, "THSP_Image", ShowMap=False, isGray=True)
    plt.subplot(122)
    ShowImage(COM, "COM", ShowMap=False, isGray=True)
    plt.show()

    # Calculating IM and AVD values
    IM, AVD = CalculateIM_AVD(Normalized_COM)

    # Finding GD (Generalized Difference Method)
    GD, PGD, AGD = GetGDs(SpeckleImages)

    # Fujii Method
    F, F_al, PGAF = GetFujiis(SpeckleImages)

    # SM Index
    SM_Index = GetSMIndex(SpeckleImages)

    # Contrast and Correlation
    Contrast, Homogeneity, AngularSecondMoment = CalculateTexturalParams(SpeckleImages)

    print("\nResults:")
    print(f"IM                  : {IM}")
    print(f"AVD                 : {AVD}")
    print(f"SM Index            : {SM_Index}")
    print(f"Contrast            : {Contrast}")
    print(f"Homogeneity         : {Homogeneity}")
    print(f"AngularSecondMoment : {AngularSecondMoment}\n")

    # Displaying the energy plots
    plt.subplot(231)
    ShowImage(GD, "GD")
    plt.subplot(232)
    ShowImage(PGD, "PGD")
    plt.subplot(233)
    ShowImage(AGD, "AGD")
    plt.subplot(234)
    ShowImage(F, "F")
    plt.subplot(235)
    ShowImage(F_al, "F_al")
    plt.subplot(236)
    ShowImage(PGAF, "PGAF")
    plt.show()

    return [THSP_Image, COM, Normalized_COM, GD, PGD, AGD, F, F_al, PGAF]


if __name__ == "__main__":
    # Parsing arguments
    args = ArgParse()

    # Reading the speckle images and processing them
    SpeckleImages = ReadImages(args["ImageDirPath"])
    ProcessSpecklePattern(SpeckleImages)
