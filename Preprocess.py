# Preprocess.py
# memasukkan library open cv kedalam project
import cv2
# memasukkan library numpy kedalam project
import numpy as np

# module level variables ##########################################################################
CAL_VAL = np.loadtxt("calibrated_value.txt")
(w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H, A_T_B, A_T_W, T_V, Xtrans,
 Ytrans) = np.loadtxt("calibrated_value.txt")
GAUSSIAN_SMOOTH_FILTER_SIZE = (int(G_S_F_W), int(G_S_F_H))  # last best = 3,3
ADAPTIVE_THRESH_BLOCK_SIZE = int(A_T_B)  # 19 , last best = 19
ADAPTIVE_THRESH_WEIGHT = int(A_T_W)  # 9, last best = 11
THRESHOLD_VALUE = int(T_V)


##
###################################################################################################
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal) # mengambil nilai pada gambar asli

    imgGrayscale = np.invert(imgGrayscale)  # menghitung bitwise pada gambar
    
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale) # memaksimalkan tingkat kecerahan pada gambar
    
    height, width = imgGrayscale.shape # mengambil nilai tinggi dan lebar dari gambar

    imgBlurred = np.zeros((height, width, 1), np.uint8) # membuat wadah untuk gambar blur setelah pemprosesan
    # cv2.imshow("c_3", imgBlurred )

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0) # proses blur dengan gaussian blur
    # cv2.imshow("imgBlurred", imgBlurred )
    # imgBlurred = np.invert(imgBlurred)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, THRESHOLD_VALUE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT) # melakukan thresholding terhadap gambar dan penambahan fungsi lain kedalam gambar
    # imgThresh = np.invert(imgThresh)
    # cv2.imshow("cobaaa", imgThresh)

    return imgGrayscale, imgThresh # pengembalian gambar grayscale dan gambar hasil thresh

# end function

###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape # mengambil nilai tinggi, lebar dan numchannel dari gambar

    imgHSV = np.zeros((height, width, 3), np.uint8) # membuat matriks pada gambar yang penuh dengan nol
 
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV) # mengubah color menjadi HSV color space.

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV) # membagi gambar multisaluran sumber menjadi beberapa gambar saluran tunggal. 
 
    return imgValue


# end function

###################################################################################################
def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8) # membuat matriks pada gambar yang penuh dengan nol
    imgBlackHat = np.zeros((height, width, 1), np.uint8) # membuat matriks pada gambar yang penuh dengan nol

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement) # menutup lubang kecil di dalam objek latar depan, atau titik hitam kecil pada objek.
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement) # menutup lubang kecil di dalam objek latar depan, atau titik hitam kecil pada objek.

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat) # menambahkan gambar hasil grayscale dengan gambar hasil top hat
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat) # membuat subtrach dari hasil gambar yang sudah di proses

    return imgGrayscalePlusTopHatMinusBlackHat
# end function
