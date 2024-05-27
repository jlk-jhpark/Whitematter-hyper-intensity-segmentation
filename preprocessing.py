import os
import pydicom
import numpy as np
import cv2
import peakutils

def read_dicom(dcm_path):
    rows_standard,cols_standard = [256,256]
    img = getRealignment_WMH(
        dcm_path, rows_standard, cols_standard
    )
    data_input = preprocessing_WMH(
        np.float32(img),  # type: ignore
        rows_standard,
        cols_standard,
    )
    return data_input
    
def getRealignment_WMH(dcm_base_path, rows_standard, cols_standard):
    imgs = []
    dcm_paths = os.listdir(dcm_base_path)
    dcm_paths.sort()
    I_num = []

    for dcm_path in dcm_paths:
        dcm = pydicom.dcmread(os.path.join(dcm_base_path, dcm_path), force=True)
        if not hasattr(dcm.file_meta,'TransferSyntaxUID'):
            dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # type: ignore
        required_elements = [
            "BitsAllocated",
            "PixelRepresentation",
            "SamplesPerPixel",
        ]

        for elem in required_elements:
            if elem not in dcm:
                if elem == "BitsAllocated":
                    dcm.BitsAllocated = 8
                if elem == "PixelRepresentation":
                    dcm.PixelRepresentation = 1
                if elem == "SamplesPerPixel":
                    dcm.SamplesPerPixel = 1

        img = dcm.pixel_array
        img = 255 * (img / np.max(img))
        img = cv2.resize(
            img, (rows_standard, cols_standard), interpolation=cv2.INTER_CUBIC
        )
        imgs.append(img)
        I_num.append(int(dcm.InstanceNumber))

    if len(imgs) == 0:
        return imgs
    if len(np.unique(I_num))>10:
        sort_idx = np.argsort(I_num)
        imgs = np.array(imgs)[sort_idx]
    else:
        sort_idx = np.argsort(I_num)
        imgs = np.array(imgs)[sort_idx]

    imgs = np.array(imgs)
    minValue = 99999999
    minIndex = 0
    imgs_flat = imgs.ravel()
    imgs_flat = np.trim_zeros(imgs_flat)
    hist, centers = np.histogram(imgs_flat, 32)
    indexes = peakutils.indexes(hist, thres=7.0 / max(hist), min_dist=1)
    for x in indexes:
        if minValue > abs(150 - centers[x]):
            minIndex = x
            minValue = abs(150 - centers[x])
        coeff = 150 / centers[minIndex]

    imgs = imgs * coeff  # type: ignore
    imgs = imgs.clip(max=255)
    imgs = imgs.astype("uint8")
    return imgs


def preprocessing_WMH(FLAIR_array, rows_standard, cols_standard):
    thresh = 30  # threshold for getting the brain mask

    mid_slice = FLAIR_array.shape[0] // 2
    brain_mask = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    brain_mask[FLAIR_array >= thresh] = 1
    brain_mask[FLAIR_array < thresh] = 0
    brain_mask[mid_slice, :, :] = scipy.ndimage.morphology.binary_fill_holes(  # type: ignore
        brain_mask[mid_slice, :, :]
    )
    FLAIR_array -= np.mean(FLAIR_array[brain_mask == 1])  # Gaussion Normalization
    FLAIR_array /= np.std(FLAIR_array[brain_mask == 1])
    #     FLAIR_array -=mean      #Gaussion Normalization
    #     FLAIR_array /=std

    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    FLAIR_array = FLAIR_array[
        :,
        int((rows_o - rows_standard) / 2) : int((rows_o - rows_standard) / 2)
        + rows_standard,
        int((cols_o - cols_standard) / 2) : int((cols_o - cols_standard) / 2)
        + cols_standard,
    ]

   
    return FLAIR_array[..., np.newaxis]