# DetectPlates.py

# memasukkan library math kedalam project
import math
# memasukkan library math kedalam project
import cv2
# memasukkan library numpy kedalam project
import numpy as np
# memasukkan file  DetectChars kedalam project
import DetectChars
# memasukkan file possiblechar kedalam project
import PossibleChar
# memasukkan file preprocess kedalam project
import Preprocess
# memasukkan file PossiblePlate kedalam project
import PossiblePlate


# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.1
PLATE_HEIGHT_PADDING_FACTOR = 1.5


# 1.3 dan 1.5

####################################################################################################

# fungsi deteksi plat pada scene
def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []  # ini akan menjadi nilai yang dikembalikan

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(
        imgOriginalScene)  # preproses untuk mendapatkan gambar skala abu-abu dan thresholds 

    # menemukan semua kemungkinan karakter di tempat kejadian,
    # fungsi ini pertama-tama menemukan semua kontur, maka hanya mencakup kontur yang bisa menjadi chars (tanpa dibandingkan dengan karakter 
    # lain belum)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    # diberikan daftar semua karakter yang mungkin, temukan kelompok karakter yang cocok
    # dalam langkah-langkah berikutnya setiap kelompok karakter yang cocok akan mencoba untuk diakui sebagai plat
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:  # untuk setiap kelompok karakter yang cocok
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)  # mencoba untuk mengekstrak plat

        if possiblePlate.imgPlate is not None:  # jika plat ditemukan
            listOfPossiblePlates.append(possiblePlate)  # tambahkan ke daftar kemungkinan plat
        # end if
    # end for

    return listOfPossiblePlates


# end function

###################################################################################################
def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []  # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)  # menemukan semua kontur
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):  # perulangan untuk setiap kontur

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(
                possibleChar):  # jika kontur adalah karakter yang mungkin, perhatikan ini tidak dibandingkan dengan karakter lain (belum)
            intCountOfPossibleChars = intCountOfPossibleChars + 1  # jumlah kenaikan dari kemungkinan chars
            listOfPossibleChars.append(possibleChar)  # dan tambahkan ke daftar karakter yang mungkin
        # end if
    # end for

    return listOfPossibleChars


# end function


###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()  # ini akan menjadi nilai yang dikembalikan

    listOfMatchingChars.sort(
        key=lambda matchingChar: matchingChar.intCenterX)  # urutkan karakter dari kiri ke kanan berdasarkan posisi x

    # menghitung titik tengah plat
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # menghitung lebar dan tinggi plat
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[
        len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[
                             0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # hitung sudut koreksi wilayah plat
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0],
                                                     listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # mengemas titik pusat wilayah pelat, lebar dan tinggi, dan sudut koreksi ke variabel plat anggota rektek yang diputar
    possiblePlate.rrLocationOfPlateInScene = (
    tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg)

    # langkah terakhir adalah melakukan rotasi aktual

    # dapatkan matriks rotasi untuk sudut koreksi terhitung kami
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape  # membongkar lebar dan tinggi gambar asli

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))  # memutar seluruh gambar

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped  # salin gambar pelat yang dipotong ke dalam variabel anggota yang berlaku dari pelat yang mungkin

    return possiblePlate
# end function
