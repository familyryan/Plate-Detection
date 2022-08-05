# GenData.py

# memasukkan library argparse kedalam project
import argparse
# memasukkan library os kedalam project
import os
# memasukkan library sys kedalam project
import sys

# memasukkan library math kedalam project
import cv2
# memasukkan library numpy kedalam project
import numpy as np
# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


###################################################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--image_train",
                    help="path for the images that you're going to train")
    args = vars(ap.parse_args())
    if args.get("image", True):
        imgTrainingNumbers = cv2.imread(args["image_train"])  # membaca  gambar dalam nomor pelatihan
        if imgTrainingNumbers is None:
            print
            "error: image not read from file \n\n"  # cetak pesan kesalahan untuk std out
            os.system("pause")  # jeda sehingga pengguna dapat melihat pesan kesalahan
            return
    else:
        print("Please add -d or --image_train argument")

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # mendapatkan gambar grayscale 
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

    # filter gambar dari skala abu-abu ke hitam dan putih
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                      255,  # membuat piksel yang melewati threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      # menggunakan gaussian daripada rata-rata, tampaknya memberikan hasil yang lebih baik
                                      cv2.THRESH_BINARY_INV,
                                      # invert sehingga latar depan akan putih, latar belakang akan hitam
                                      11,  # ukuran lingkungan piksel yang digunakan untuk menghitung nilai threshold
                                      2)  # konstanta dikurangi dari rata-rata atau rata-rata tertimbang

    cv2.imshow("imgThresh", imgThresh)  # perlihatkan gambar ambang batas untuk referensi

    imgThreshCopy = imgThresh.copy()  # membuat salinan gambar thresh, ini dalam findContours b / c yang diperlukan memodifikasi gambar

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              # input image, pastikan untuk menggunakan salinan karena fungsi akan memodifikasi gambar ini dalam rangka menemukan kontur
                                                              cv2.RETR_EXTERNAL,  # ambil kontur terluar saja
                                                              cv2.CHAIN_APPROX_SIMPLE)  # kompres segmen horizontal, vertikal, dan diagonal dan hanya menyisakan titik akhir

    # menyatakan array numpy kosong, kita akan menggunakan ini untuk menulis ke file nanti
    # baris nol, kol yang cukup untuk menyimpan semua data gambar
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []  # menyatakan daftar klasifikasi kosong, ini akan menjadi daftar kami tentang bagaimana kami mengklasifikasikan karakter kami dari input pengguna, kami akan menulis ke file di akhir

    # kemungkinan karakter yang kami minati adalah digit 0 hingga 9, masukkan ini ke dalam daftar intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
                     ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),
                     ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z')]
    for npaContour in npaContours:  # untuk setiap kontur
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:  # jika kontur cukup besar untuk dipertimbangkan
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # dapatkan dan pecahkan rect bounding rect

            # menggambar persegi panjang di sekitar setiap kontur saat kami meminta masukan kepada pengguna
            cv2.rectangle(imgTrainingNumbers,  # gambar persegi panjang pada gambar pelatihan asli
                          (intX, intY),  # sudut kiri atas
                          (intX + intW, intY + intH),  # sudut kanan bawah
                          (0, 0, 255),  # red
                          2)  # thickness

            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]  # potong char dari gambar threshold 
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                                RESIZED_IMAGE_HEIGHT))  # mengubah ukuran gambar, ini akan lebih konsisten untuk pengakuan dan penyimpanan

            cv2.imshow("imgROI", imgROI)  # perlihatkan karakter yang dipotong untuk referensi
            cv2.imshow("imgROIResized", imgROIResized)  # perlihatkan gambar yang di ukurannya untuk referensi
            cv2.imshow("training_numbers.png",
                       imgTrainingNumbers)  # tampilkan gambar angka pelatihan, ini sekarang akan memiliki persegi panjang merah yang digambar di atasnya

            intChar = cv2.waitKey(0)  # dapatkan penekanan tombol

            if intChar == 27:  # jika tombol ESC ditekan
                sys.exit()  # exit program
            elif intChar in intValidChars:  # lain jika char ada dalam daftar chars yang kita cari . . .

                intClassifications.append(
                    intChar)  # tambahkan char klasifikasi ke daftar karakter bilangan bulat (kami akan dikonversi ke mengambang nanti sebelum menulis ke file)

                npaFlattenedImage = imgROIResized.reshape((1,
                                                           RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # ratakan gambar ke array numpy 1d sehingga kita dapat menulis untuk file nanti
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage,
                                               0)  # tambahkan array numpy impage pipih saat ini ke daftar array numpy gambar pipih
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications,
                                  np.float32)  # mengonversi daftar klasifikasi ints ke array numpy floats

    npaClassifications = fltClassifications.reshape(
        (fltClassifications.size, 1))  # ratakan array numpy mengapung ke 1d sehingga kita dapat menulis untuk file nanti

    print
    "\n\ntraining complete !!\n"

    np.savetxt("classifications.txt", npaClassifications)  # tulis gambar pipih ke berkas
    np.savetxt("flattened_images.txt", npaFlattenedImages)
    changeCaption()  #

    cv2.destroyAllWindows()  # remove windows from memory

    return


###################################################################################################
def changeCaption():
    data = np.loadtxt("classifications.txt")
    i = 0
    for a in data:
        a = int(round(a))
        if (a == ord('a')):
            data[i] = ord('A')
        if (a == ord('b')):
            data[i] = ord('B')
        if (a == ord('c')):
            data[i] = ord('C')
        if (a == ord('d')):
            data[i] = ord('D')
        if (a == ord('e')):
            data[i] = ord('E')
        if (a == ord('f')):
            data[i] = ord('F')
        if (a == ord('g')):
            data[i] = ord('G')
        if (a == ord('h')):
            data[i] = ord('H')
        if (a == ord('i')):
            data[i] = ord('I')
        if (a == ord('j')):
            data[i] = ord('J')
        if (a == ord('k')):
            data[i] = ord('K')
        if (a == ord('l')):
            data[i] = ord('L')
        if (a == ord('m')):
            data[i] = ord('M')
        if (a == ord('n')):
            data[i] = ord('N')
        if (a == ord('o')):
            data[i] = ord('O')
        if (a == ord('p')):
            data[i] = ord('P')
        if (a == ord('q')):
            data[i] = ord('Q')
        if (a == ord('r')):
            data[i] = ord('R')
        if (a == ord('s')):
            data[i] = ord('S')
        if (a == ord('t')):
            data[i] = ord('T')
        if (a == ord('u')):
            data[i] = ord('U')
        if (a == ord('v')):
            data[i] = ord('V')
        if (a == ord('w')):
            data[i] = ord('W')
        if (a == ord('x')):
            data[i] = ord('X')
        if (a == ord('y')):
            data[i] = ord('Y')
        if (a == ord('z')):
            data[i] = ord('Z')
        i = i + 1

    
    hasil = np.array(data, np.float32)  # mengonversi daftar klasifikasi ints ke array numpy floats
    npaClassifications = hasil.reshape((hasil.size, 1))

    np.savetxt("classifications.txt", npaClassifications)
    # print("char was change to caption !")


if __name__ == "__main__":
    main()
# end if
