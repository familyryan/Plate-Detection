# DetectChars.py
# memasukkan library math kedalam project
import math
# memasukkan library math kedalam project
import os
# memasukkan library math kedalam project
import cv2
# memasukkan library numpy kedalam project
import numpy as np
# memasukkan file main kedalam project
import Main
# memasukkan file possiblechar kedalam project
import PossibleChar
# memasukkan file preprocess kedalam project
import Preprocess

# module level variables ##########################################################################

kNearest = cv2.ml.KNearest_create()

# konstanta untuk checkIfPossibleChar, ini hanya memeriksa satu karakter yang mungkin (tidak dibandingkan dengan karakter lain)
MIN_PIXEL_WIDTH = 2  # 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25  # 0.25
MAX_ASPECT_RATIO = 1.0  # 1.0

MIN_PIXEL_AREA = 80

# konstanta untuk membandingkan dua karakter
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.01
MAX_DIAG_SIZE_MULTIPLE_AWAY = 8.0

MAX_CHANGE_IN_AREA = 0.5  # 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0  # 12

# konstanta lainnya
MIN_NUMBER_OF_MATCHING_CHARS = 5

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


def loadKNNDataAndTrainKNN():
    allContoursWithData = []  # mendeklarasikan daftar kosong,
    validContoursWithData = []  # kita akan mengisi ini segera

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)  # baca dalam klasifikasi pelatihan
    except:  # jika berkas tak bisa dibuka
        print("error, unable to open classifications.txt, exiting program\n")  # show error message
        os.system("pause")
        return False  # and return False
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # read in training images
    except:  # if file could not be opened
        print("error, unable to open flattened_images.txt, exiting program\n")  # show error message
        os.system("pause")
        return False  # and return False
    # end try

    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))  # membentuk kembali array numpy ke 1d, diperlukan untuk lulus untuk menelepon ke train

    kNearest.setDefaultK(1)  # set default K to 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)  # train KNN object

    return True  # if we got here training was successful so return true


# fungsi detect chars pada plates
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:  # if list of possible plates is empty
        return listOfPossiblePlates  # return
    # end if
    # pada titik ini kita dapat yakin daftar kemungkinan piring memiliki setidaknya satu piring

    for possiblePlate in listOfPossiblePlates:  # untuk setiap plat yang mungkin, ini adalah besar untuk loop yang mengambil sebagian besar fungsi

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(
            possiblePlate.imgPlate)  # praproses untuk mendapatkan skala abu-abu dan threshold pada gambar
        # menambah ukuran gambar plat agar lebih mudah dilihat dan deteksi karakter
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx=1.7, fy=1.7)  # 1.6,1.6

        # threshold lagi untuk menghilangkan area abu-abu
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0,
                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # temukan semua kemungkinan karakter di piring,
        # fungsi ini pertama-tama menemukan semua kontur, maka hanya mencakup kontur yang bisa menjadi chars (tanpa dibandingkan dengan karakter lain belum)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        # memberikan daftar semua karakter yang mungkin, temukan kelompok karakter yang cocok di dalam plat
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if (len(listOfListsOfMatchingCharsInPlate) == 0):  # if no groups of matching chars were found in the plate

            possiblePlate.strChars = ""
            continue  # kembali ke atas untuk melakukan perulangan
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):  # dalam setiap daftar karakter yang cocok
            listOfListsOfMatchingCharsInPlate[i].sort(
                key=lambda matchingChar: matchingChar.intCenterX)  # mengurutkan karakter dari kiri ke kanan
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(
                listOfListsOfMatchingCharsInPlate[i])  # dan hapus karakter yang tumpang tindih dalam
        # end for

        # alam setiap plat yang mungkin, misalkan daftar terpanjang dari karakter pencocokan potensial adalah daftar karakter yang sebenarnya
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        # perulangan melalui semua vektor chars yang cocok, dapatkan indeks yang paling banyak
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

        # misalkan bahwa daftar terpanjang dari chars yang cocok di dalam piring adalah daftar karakter yang sebenarnya
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

    # akhir dari besar untuk loop yang mengambil sebagian besar fungsi

    return listOfPossiblePlates

# fungsi untuk mencari  karakter pada plat
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []  # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

    # mencari semua contours pada plat
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:  # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(
                possibleChar):  # jika kontur adalah karakter yang mungkin, perhatikan ini tidak dibandingkan dengan karakter lain (belum) . . .
            listOfPossibleChars.append(possibleChar)  # menambahkan kedalam list dari possible chars
        # end if
    # end if

    return listOfPossibleChars

# fungsi untuk mengecek kemungkinan karakter
def checkIfPossibleChar(possibleChar):
    # fungsi ini adalah 'first pass' yang melakukan pemeriksaan kasar pada kontur untuk melihat apakah itu bisa menjadi char,
    # perhatikan bahwa kita belum (belum) membandingkan karakter dengan karakter lain untuk mencari grup
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


def findListOfListsOfMatchingChars(listOfPossibleChars):
    # dengan fungsi ini, kita mulai dengan semua karakter yang mungkin dalam satu daftar besar
    # tujuan dari fungsi ini adalah untuk menyusun ulang satu daftar besar karakter ke dalam daftar daftar karakter yang cocok,
    # perhatikan bahwa chars yang tidak ditemukan berada dalam sekelompok pertandingan tidak perlu dipertimbangkan lebih lanjut
    listOfListsOfMatchingChars = []  # ini akan menjadi nilai yang dikembalikan

    for possibleChar in listOfPossibleChars:  # untuk setiap karakter yang mungkin dalam satu daftar besar karakter
        listOfMatchingChars = findListOfMatchingChars(possibleChar,
                                                      listOfPossibleChars)  # temukan semua karakter dalam daftar besar yang cocok dengan karakter saat ini

        listOfMatchingChars.append(possibleChar)  # juga tambahkan karakter saat ini ke daftar karakter yang cocok saat ini

        if len(
                listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:  # jika saat ini kemungkinan daftar karakter yang cocok tidak cukup lama untuk merupakan kemungkinan plat
            continue  # melompat kembali ke bagian atas untuk loop dan mencoba lagi dengan karakter berikutnya, perhatikan bahwa itu tidak perlu
            # untuk menyimpan daftar dengan cara apa pun karena tidak memiliki cukup karakter untuk menjadi piring yang mungkin
        

        # jika kita sampai di sini, daftar saat ini lulus tes sebagai "grup" atau "cluster" dari chars yang cocok
        listOfListsOfMatchingChars.append(listOfMatchingChars)  # jadi tambahkan ke daftar daftar karakter yang cocok

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        # hapus daftar karakter yang cocok saat ini dari daftar besar sehingga kami tidak menggunakan karakter yang sama dua kali,
        # pastikan untuk membuat daftar besar baru untuk ini karena kami tidak ingin mengubah daftar besar asli
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(
            listOfPossibleCharsWithCurrentMatchesRemoved)  # recursive call

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:  # untuk setiap daftar karakter yang cocok yang ditemukan oleh  recursive call
            listOfListsOfMatchingChars.append(
                recursiveListOfMatchingChars)  # tambahkan ke daftar asli daftar karakter yang cocok
        # end for

        break  # exit for

        # end for
        print(listOfListsOfMatchingChars)
    return listOfListsOfMatchingChars


def findListOfMatchingChars(possibleChar, listOfChars):
    # tujuan dari fungsi ini adalah, mengingat kemungkinan char dan daftar besar kemungkinan chars,
    # temukan semua karakter dalam daftar besar yang cocok untuk satu karakter yang mungkin, dan kembalikan karakter yang cocok sebagai daftar
    listOfMatchingChars = []  # ini akan menjadi nilai yang dikembalikan

    for possibleMatchingChar in listOfChars:  # untuk setiap karakter dalam daftar besar
        if possibleMatchingChar == possibleChar:  # jika char kami mencoba untuk menemukan kecocokan untuk adalah char yang sama persis dengan char dalam daftar besar yang saat ini kami periksa
            # maka kita tidak boleh memasukkannya dalam daftar pertandingan b / c yang akan berakhir ganda termasuk char saat ini
            continue  # jadi jangan tambahkan ke daftar kecocokan dan lompat kembali ke atas untuk loop
        # end if
        # menghitung hal-hal untuk melihat apakah chars cocok
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(
            abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(
            possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(
            abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(
            possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(
            abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(
            possibleChar.intBoundingRectHeight)

        # memeeriksa apakah karakter cocok
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
                fltChangeInArea < MAX_CHANGE_IN_AREA and
                fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
                fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
            listOfMatchingChars.append(
                possibleMatchingChar)  # jika karakter cocok, tambahkan karakter saat ini ke daftar karakter yang cocok
        # end if
    # end for

    return listOfMatchingChars  # return result


# menggunakan Pythagorean theorem untuk menghitung jarak antara dua karakter
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))


# menggunakan trigonometry (SOH CAH TOA) untuk menghitung sudut antar karakter
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:  # periksa untuk memastikan kami tidak membagi dengan nol jika posisi X tengah sama, pembagian float dengan nol akan menyebabkan crash di Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)  # if adjacent is not zero, calculate angle
    else:
        fltAngleInRad = 1.5708  # jika berdekatan adalah nol, gunakan ini sebagai sudut, ini harus konsisten dengan versi C++ dari program ini
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)  # menghitung sudut dalam derajat

    return fltAngleInDeg


# jika kita memiliki dua karakter yang tumpang tindih atau untuk menutup satu sama lain untuk mungkin menjadi karakter terpisah, lepaskan karakter bagian dalam (lebih kecil),
# ini untuk mencegah termasuk char yang sama dua kali jika dua kontur ditemukan untuk char yang sama,
# misalnya untuk huruf 'O' cincin bagian dalam dan cincin luar dapat ditemukan sebagai kontur, tetapi kita hanya harus memasukkan char sekali
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)  # ini akan menjadi nilai yang dikembalikan

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:  # jika char saat ini dan karakter lain tidak sama char . . . .
                # jika char saat ini dan char lain memiliki titik tengah di lokasi yang hampir sama . . .
                if distanceBetweenChars(currentChar, otherChar) < (
                        currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # Jika kita sampai di sini kita telah menemukan karakter tumpang tindih
                    # selanjutnya kita mengidentifikasi karakter mana yang lebih kecil, maka jika karakter itu belum dihapus pada lulus 
                    # sebelumnya, hapus
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:  # jika karakter saat ini lebih kecil dari char 
                                                                                         # lainnya
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:  # jika karakter saat ini belum dihapus pada pass 
                                                                                    # sebelumnya . . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)  # lalu hapus karakter saat ini
                    else:  # lain jika karakter lain lebih kecil dari char saat ini
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:  # jika karakter lain belum dihapus pada pass sebelumnya . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)  # lalu hapus karakter lain

    return listOfMatchingCharsWithInnerCharRemoved


# di sinilah kita menerapkan pengenalan karakter yang sebenarnya
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""  # ini akan menjadi nilai yang dikembalikan, karakter di  lic plat

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.intCenterX)  # mengurutkan karakter dari kiri ke kanan

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR,
                 imgThreshColor)  # membuat versi warna gambar ambang batas sehingga kita dapat menggambar kontur warna di atasnya

    for currentChar in listOfMatchingChars:  # untuk setiap char di piring
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth),
               (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)  # gambar kotak hijau di sekitar char

        # potong char dari gambar threshold 
        imgROI = imgThresh[
                 currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                 currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (
        RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))  # mengubah ukuran gambar, ini diperlukan untuk pengenalan karakter

        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))  # ratakan gambar ke dalam array numpy 1d

        npaROIResized = np.float32(npaROIResized)  # konversi dari array 1d numpy dari ints ke array numpy 1d floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                     k=1)  # akhirnya kita bisa memanggil FindNearest !!!

        strCurrentChar = str(chr(int(npaResults[0][0])))  # dapatkan karakter dari hasil

        strChars = strChars + strCurrentChar  # tambahkan karakter saat ini ke string penuh

    # end for

    return strChars
# end function
