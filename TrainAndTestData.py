# TrainAndTest.py

# memasukkan libray argparse kedalam project
import argparse
# memasukkan libray operator kedalam project
import operator
# memasukkan libray os kedalam project
import os
# memasukkan libray open cv kedalam project
import cv2
# memasukkan libray numpy kedalam project
import numpy as np

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


###################################################################################################
class ContourWithData():
    # member variables ############################################################################
    npaContour = None  # contour
    boundingRect = None  #garis lurus untuk kontur
    intRectX = 0  # pembatas persegi pojok kiri atas lokasi x
    intRectY = 0  # pembatas persegi pojok kiri atas y lokasi
    intRectWidth = 0  # lebar persegi pembatas
    intRectHeight = 0  # tinggi persegi pembatas
    fltArea = 0.0  # area kontur

    def calculateRectTopLeftPointAndWidthAndHeight(self):  # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):  
        if self.fltArea < MIN_CONTOUR_AREA: return False  # pemeriksaan validitas yang jauh lebih baik akan diperlukan
        return True


###################################################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--image_testing",
                    help="path for the images that you're going to test")
    args = vars(ap.parse_args())
    if args.get("image", True):
        imgTestingNumbers = cv2.imread(args["image_testing"])  # baca dalam gambar nomor pelatihan
        if imgTestingNumbers is None:
            print("error: image not read from file \n\n")  # cetak pesan kesalahan untuk std out
            os.system("pause")  # jeda sehingga pengguna dapat melihat pesan kesalahan
            return
    else:
        print("Please add -d or --image_testing argument")

    allContoursWithData = []  # menyatakan daftar kosong,
    validContoursWithData = []  # kita akan mengisi ini segera

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)  # membaca pada training classifications
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # membaca gambar training 
    except
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))  # membentuk kembali array numpy ke 1d, diperlukan untuk lulus untuk menelepon ke kereta

    kNearest = cv2.ml.KNearest_create()  # instantiate objek KNN

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)  # mendapatkan gambar grayscale
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

    # filter gambar dari skala abu-abu ke hitam dan putih
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                      255,  # membuat piksel yang melewati ambang batas full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      # gunakan gaussian daripada rata-rata, tampaknya memberikan hasil yang lebih baik
                                      cv2.THRESH_BINARY_INV,
                                      # invert sehingga latar depan akan putih, latar belakang akan hitam
                                      11,  # ukuran lingkungan piksel yang digunakan untuk menghitung nilai ambang batas
                                      2)  # konstanta dikurangi dari rata-rata atau rata-rata tertimbang

    imgThreshCopy = imgThresh.copy()  # membuat salinan gambar thresh, ini dalam findContours b / c yang diperlukan memodifikasi gambar

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              # input image, pastikan untuk menggunakan salinan karena fungsi akan memodifikasi gambar ini dalam rangka menemukan kontur
                                                              cv2.RETR_EXTERNAL,  # ambil kontur terluar saja
                                                              cv2.CHAIN_APPROX_SIMPLE)  # kompres segmen horizontal, vertikal, dan diagonal dan hanya menyisakan titik akhir

    for npaContour in npaContours:  # untuk setiap kontur
        contourWithData = ContourWithData()  # instantiate kontur dengan objek data
        contourWithData.npaContour = npaContour  # tetapkan kontur ke kontur dengan data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)  # mendapatkan perbaiki terikat
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()  # dapatkan info rect yang terikat
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)  # menghitung area kontur
        allContoursWithData.append(contourWithData)  # menambahkan kontur dengan objek data ke daftar semua kontur dengan data
    # end for

    for contourWithData in allContoursWithData:  # untuk semua kontur
        if contourWithData.checkIfContourIsValid():  # periksa apakah valid
            validContoursWithData.append(contourWithData)  # jika demikian, tambahkan ke daftar kontur yang valid
        # end if
    # end for

    validContoursWithData.sort(key=operator.attrgetter("intRectX"))  # urutkan kontur dari kiri ke kanan

    strFinalString = ""  # menyatakan string akhir, ini akan memiliki urutan angka akhir pada akhir program

    for contourWithData in validContoursWithData:  # for each contour
        # menggambar sebuah rect hijau di sekitar char saat ini
        cv2.rectangle(imgTestingNumbers,  # menggambar persegi panjang pada gambar pengujian asli
                      (contourWithData.intRectX, contourWithData.intRectY),  # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth,
                       contourWithData.intRectY + contourWithData.intRectHeight),  # lower right corner
                      (0, 255, 0),  # green
                      2)  # thickness

        imgROI = imgThresh[contourWithData.intRectY: contourWithData.intRectY + contourWithData.intRectHeight,
                 # potong char dari gambar ambang batas
                 contourWithData.intRectX: contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                            RESIZED_IMAGE_HEIGHT))  # mengubah ukuran gambar, ini akan lebih konsisten untuk pengakuan dan penyimpanan

        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # ratakan gambar ke dalam array numpy 1d

        npaROIResized = np.float32(npaROIResized)  # konversi dari array 1d numpy dari ints ke array numpy 1d floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                     k=1)  # memanggil fungsi KNN find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))  # mendapatkan karakter dari hasil

        strFinalString = strFinalString + strCurrentChar  # menambahkan karakter saat ini ke string penuh
    # end for

    print("\n" + strFinalString + "\n")  # menampilkan string lengkap

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)  # perlihatkan gambar input dengan kotak hijau yang digambar di sekitar digit yang ditemukan
    cv2.waitKey(0)  # wait for user key press

    cv2.destroyAllWindows()  # hapus jendela dari memori

    return


###################################################################################################
if __name__ == "__main__":
    main()
# end if
