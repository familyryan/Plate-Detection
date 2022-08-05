# USAGE
# python Main.py --image Sample/s1.jpg  @ untuk file gambar
# python Main.py --video Sample/sv1.mp4  @ untuk file video
# python Main.py @ untuk cam

# memasukkan library argparse kedalam project
import argparse
# memasukkan library os kedalam project
import os
# memasukkan library open cv kedalam project
import cv2
# memasukkan file project calibrartion kedalam project ini
import Calibration as cal
# memasukkan file project calibrartion kedalam project ini
import DetectChars
# memasukkan file project DetectChars kedalam project ini
import DetectPlates
# memasukkan file project Preprocess kedalam project ini
import Preprocess as pp
# memasukkan file project imutils kedalam project ini
import imutils

# Module level variables for image

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
N_VERIFY = 5  # number of verification

# fungsi main


def main():
    # argument untuk input video/image/calibration
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to video file")
    ap.add_argument("-i", "--image", help="Path to the image")
    ap.add_argument("-c", "--calibration", help="image or video or camera")
    args = vars(ap.parse_args())

    img_original_scene = None
    loop = None
    camera = None

    # jika -c ditetapkan, kalibrasi sudut kamera atau video
    if args.get("calibration", True):
        img_original_scene = cv2.imread(
            args["calibration"])  # membaca fungsi kalibrasi
        if img_original_scene is None:
            print("Please check again the path of image or argument !")
        img_original_scene = imutils.resize(
            img_original_scene, width=720)  # pembentukan ukuran gambar
        cal.calibration(img_original_scene)
        return
    else:  # membuka video / image / cam
        if args.get("video", True):
            camera = cv2.VideoCapture(args["video"])
            if camera is None:
                print("Please check again the path of video or argument !")
            loop = True

        elif args.get("image", True):
            img_original_scene = cv2.imread(args["image"])
            if img_original_scene is None:
                print("Please check again the path of image or argument !")
                loop = False
        else:
            camera = cv2.VideoCapture(0)
            loop = True

    # memuat dan mengecek KNN Model
    assert DetectChars.loadKNNDataAndTrainKNN(), "KNN can't be loaded !"

    save_number = 0
    prev_license = ""
    licenses_verify = []

    # perulangan untuk Video
    while loop:
        # ambil bingkai saat ini
        (grabbed, frame) = camera.read()
        if args.get("video") and not grabbed:
            break

        # mengubah ukuran bingkai dan praproses
        img_original_scene = imutils.resize(frame, width=620)
        _, img_thresh = pp.preprocess(img_original_scene)

        # menampilkan hasil praproses
        cv2.imshow("threshold", img_thresh)

        # mendapatkan lisensi dalam bingkai
        img_original_scene = imutils.transform(img_original_scene)
        img_original_scene, new_license = searching(img_original_scene, loop)

        # hanya menyimpan 5 lisensi yang sama setiap kali (verifikasi)
        if new_license == "":
            print("no characters were detected\n")
        else:
            if len(licenses_verify) == N_VERIFY and len(set(licenses_verify)) == 1:
                if prev_license == new_license:
                    print(f"still = {prev_license}\n")
                else:
                    # menampilkan dan simpan pelat terverifikasi
                    print(
                        f"A new license plate read from image = {new_license} \n")
                    # menampilkan hasil dari license
                    cv2.imshow(new_license, img_original_scene)
                    # menamai file hasil dari license
                    file_name = f"hasil/{new_license}.png"
                    # menulis ulang file license
                    cv2.imwrite(file_name, img_original_scene)
                    prev_license = new_license  # membuat variabel baru untuk license
                    licenses_verify = []  # membuat wadah array untuk license
            else:
                if len(licenses_verify) == N_VERIFY:
                    # drop pertama jika mencapai N_VERIFY
                    licenses_verify = licenses_verify[1:]
                licenses_verify.append(new_license)

        # menambahkan teks dan persegi panjang, hanya untuk informasi dan batas
        cv2.putText(img_original_scene, "Press 's' to save frame to be 'save.png', for calibrating", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, bottomLeftOrigin=False)
        cv2.rectangle(img_original_scene,
                      ((img_original_scene.shape[1] // 2 - 230),
                       (img_original_scene.shape[0] // 2 - 80)),
                      ((img_original_scene.shape[1] // 2 + 230),
                       (img_original_scene.shape[0] // 2 + 80)), SCALAR_GREEN,
                      3)
        cv2.imshow("imgOriginalScene", img_original_scene)

        key = cv2.waitKey(5) & 0xFF
        # jika tombol 's' ditekan simpan gambar
        if key == ord('s'):
            save_number = str(save_number)
            savefileimg = "calib_knn/img_" + save_number + \
                ".png"  # mengesave gambar hasil proses
            savefileThr = "calib_knn/Thr_" + save_number + \
                ".png"  # menyimpan hasil gambar threshold
            # menulis ulang file gambar hasil prose
            cv2.imwrite(savefileimg, frame)
            # menulis ulang hasil gambar threshold
            cv2.imwrite(savefileThr, img_thresh)
            print("image save !")
            save_number = int(save_number)  # menyimpan nomor
            save_number = save_number + 1  # perulangan penyimpanan nomor
        if key == 27:  # jika tombol 'q' ditekan, hentikan loop
            camera.release()  # membersihkan kamera dan tutup jendela yang terbuka
            break

    # perulangan untuk gambar
    if not loop:
        img_original_scene = imutils.resize(img_original_scene, width=720)
        cv2.imshow("original", img_original_scene)
        imgGrayscale, img_thresh = pp.preprocess(img_original_scene)
        cv2.imshow("threshold", img_thresh)
        cv2.imwrite("threshold.png", img_thresh)
        img_original_scene = imutils.transform(img_original_scene)
        img_original_scene, new_license = searching(img_original_scene, loop)
        print(f"license plate read from image = {new_license} \n")
        cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# fungsi menggambar kotak pada plat


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    # mendapatkan 4 simpul dari rect yang diputar
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(
        p2fRectPoints[1]), SCALAR_RED, 2)  # mengambar 4 garis merah
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(
        p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(
        p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(
        p2fRectPoints[0]), SCALAR_RED, 2)

# fungsi untuk menebak karakter yang terdapat pada plat


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0  # ini akan menjadi pusat area teks akan ditulis untuk plat
    ptCenterOfTextAreaY = 0

    # ini akan menjadi kiri bawah area yang teks akan ditulis untuk plat
    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # pilih font jane biasa
    # skala font dasar pada tinggi area plat
    fltFontScale = float(plateHeight) / 30.0
    # ketebalan font dasar pada skala font
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)  # panggil getTextSize

    # membongkar perbaiki ke titik tengah, lebar dan tinggi, dan sudut
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    # pastikan nilai tengah adalah bilangan bulat
    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    # lokasi horizontal area teks sama dengan plat
    ptCenterOfTextAreaX = int(intPlateCenterX)

    # jika pelat nomor berada di atas 3/4 gambar
    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  # tulis karakter di bawah plat
    else:  # lain jika plat nomor berada di 1/4 bawah gambar
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))  # menulis karakter di atas plat
    # end if

    # membongkar lebar dan tinggi ukuran teks
    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))  # menghitung asal kiri bawah area teks
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))  # berdasarkan pusat, lebar, dan tinggi area teks

    # tulis teks pada gambar
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)

# fungsi pencarian


def searching(imgOriginalScene, loop):
    licenses = ""
    if imgOriginalScene is None:  # jika gambar tidak berhasil dibaca
        # cetak pesan kesalahan untuk std out
        print("error: image not read from file \n")
        # jeda sehingga pengguna dapat melihat pesan kesalahan
        os.system("pause")
        return
        # end if

    # mendeteksi plat
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
    # mendeteksi karakter dalam plat
    listOfPossiblePlates = DetectChars.detectCharsInPlates(
        listOfPossiblePlates)

    if not loop:
        cv2.imshow("imgOriginalScene", imgOriginalScene)

    if len(listOfPossiblePlates) == 0:
        if not loop:  # jika tidak ada plat yang ditemukan
            # menginformasikan pengguna tidak ada plat yang ditemukan
            print("no license plates were detected\n")
    else:  # else
        # jika kita masuk ke sini daftar kemungkinan piring memiliki di leat satu plat

        # mengurutkan daftar pelat yang mungkin dalam urutan MENURUN (sebagian besar jumlah karakter ke jumlah paling sedikit chars)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(
            possiblePlate.strChars), reverse=True)
        # misalkan piring dengan chars yang paling dikenal (piring pertama diurutkan dengan panjang string menurun
        # order) adalah plat yang sebenarnya
        licPlate = listOfPossiblePlates[0]

        if not loop:
            # tampilkan pemotongan plat
            cv2.imshow("imgPlate", licPlate.imgPlate)
            # menampilkan hasil gambar plat setelah threshold
            cv2.imshow("imgThresh", licPlate.imgThresh)
            cv2.imwrite("imgPlate.png", licPlate.imgPlate)
            cv2.imwrite("imgThresh.png", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:  # jika tidak ada karakter yang ditemukan di plat
            if not loop:
                print("no characters were detected\n")
                return  # show message
            # end if
        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
        licenses = licPlate.strChars

        if not loop:
            # tulis teks pelat nomor untuk std out
            print("license plate read from image = " + licPlate.strChars + "\n")
            # tulis teks pelat nomor pada gambar

        if not loop:
            # tampilkan ulang gambar adegan
            cv2.imshow("imgOriginalScene", imgOriginalScene)
            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

    return imgOriginalScene, licenses


if __name__ == "__main__":
    main()
