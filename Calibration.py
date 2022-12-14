# memasukkan library open cv 2 kedalam project
import cv2
# memasukkan library numpy kedalam project
import numpy as np

# memasukkan file DetectChars kedalam project
import DetectChars
# memasukkan file main kedalam project
import Main
# memasukkan file preprocess kedalam project
import Preprocess as pp
# memasukkan library imutils kedalam project
import imutils

# membuat fungsi nothing 
def nothing(x):
    pass

# membuat fungsi calibration untuk gambar
def calibration(image):
    # deklarasi window untuk posisi Calibrating pada gambar
    WindowName1 = "Calibrating Position of image"
    # deklarasi window untuk Color Thresholding
    WindowName2 = "Color Thresholding"
    # deklarasi window untuk Calibrating untuk Preprocess
    WindowName3 = "Calibrating for Preprocess"

    # membuat window berdasarkan keperluan yang sebelumnya di deklarasi
    cv2.namedWindow(WindowName2)
    cv2.namedWindow(WindowName3)
    cv2.namedWindow(WindowName1)

    # memuat data yang sudah di save dari nilai calibrated 
    (w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H, A_T_B, A_T_W, T_V, Xtrans,
     Ytrans) = np.loadtxt("calibrated_value.txt")

    # mengubah data yang sebelumnya di muat menjadi wadah baru  
    Xtrans = int(round(Xtrans + 100))
    Ytrans = int(round(Ytrans + 100))
    xValue = int(round(100 - (rotationx * 20000.0)))
    yValue = int(round((rotationy * 20000.0) + 100))
    zValue = int(round(100 - (rotationz * 100)))
    wValue = int(round(100 - ((dist - 1.0) * 200.0)))
    dValue = int(round((stretchX - 1.0) * -200.0 + 100))

    #  membuat trackbar Calibrating Position of image
    cv2.createTrackbar('Xtrans', WindowName1, Xtrans, 200, nothing)  #  untuk rotasi pada x axis
    cv2.createTrackbar('Ytrans', WindowName1, Ytrans, 200, nothing)  # untuk rotasi pada x axis
    cv2.createTrackbar("Xrot", WindowName1, xValue, 200, nothing)  # untuk rotasi pada x axis
    cv2.createTrackbar("Yrot", WindowName1, yValue, 200, nothing)  # untuk rotasi pada y axis
    cv2.createTrackbar("Zrot", WindowName1, zValue, 200, nothing)  # untuk rotasi pada z axis
    cv2.createTrackbar("ZOOM", WindowName1, wValue, 200, nothing)  # untuk fitur zoom pada gambar 
    cv2.createTrackbar("Strech", WindowName1, dValue, 200, nothing)  # untuk meregang gambar pada x axis

    switch = '0 : OFF \n1 : ON'
    
  #  membuat trackbar  Calibrating for Preprocess
    cv2.createTrackbar(switch, WindowName3, 0, 1,
                       nothing)  # beralih untuk melihat preprocess threshold , untuk detail lebih lanjut Preprocess.py
    cv2.createTrackbar('G_S_F_W', WindowName3, int(G_S_F_W), 50, nothing)  # GAUSSIAN_SMOOTH_FILTER_SIZE_WEIGHT
    cv2.createTrackbar('G_S_F_H', WindowName3, int(G_S_F_H), 50, nothing)  # GAUSSIAN_SMOOTH_FILTER_SIZE_HEIGHT
    cv2.createTrackbar('A_T_B', WindowName3, int(A_T_B), 50, nothing)  # ADAPTIVE_THRESH_BLOCK_SIZE
    cv2.createTrackbar('A_T_W', WindowName3, int(A_T_W), 50, nothing)  # ADAPTIVE_THRESH_WEIGHT
    cv2.createTrackbar('T_V', WindowName3, int(T_V), 255, nothing)  # THRESHOLD_VALUE
 #  membuat trackbar Color Thresholding
    cv2.createTrackbar("RGBSwitch", WindowName2, 0, 1, nothing)
    cv2.createTrackbar('Ru', WindowName2, 255, 255, nothing)
    cv2.createTrackbar('Gu', WindowName2, 255, 255, nothing)
    cv2.createTrackbar('Bu', WindowName2, 255, 255, nothing)
 #  membuat trackbar Color Thresholding
    cv2.createTrackbar('Rl', WindowName2, 0, 255, nothing)
    cv2.createTrackbar('Gl', WindowName2, 0, 255, nothing)
    cv2.createTrackbar('Bl', WindowName2, 50, 255, nothing)

    # mengalokasi tujuan dari gambar 
    backGround1 = np.ones((100, 500))
    backGround2 = np.ones((100, 500))
    backGround3 = np.ones((100, 500))
    # Loop untuk mendapatkan trackbar pos dan memprosesnya

    while True:
        # mendapatkan posisi di trackbar untuk transformasi perubahan
        Xtrans = cv2.getTrackbarPos('Xtrans', WindowName1)
        Ytrans = cv2.getTrackbarPos('Ytrans', WindowName1)
        X = cv2.getTrackbarPos("Xrot", WindowName1)
        Y = cv2.getTrackbarPos("Yrot", WindowName1)
        Z = cv2.getTrackbarPos("Zrot", WindowName1)
        W = cv2.getTrackbarPos("ZOOM", WindowName1)
        D = cv2.getTrackbarPos("Strech", WindowName1)

        # mendapatkan posisi di trackbar untuk switch
        S = cv2.getTrackbarPos(switch, WindowName3)  # switch untuk melihat kalibrasi threshold 

        # mendapatkan nilai dari tracbar dan membuat ood dan nilai lebih dari 3 untuk mengkalibrasi threshold 
        G_S_F_W = makeood(cv2.getTrackbarPos('G_S_F_W', WindowName3))
        G_S_F_H = makeood(cv2.getTrackbarPos('G_S_F_H', WindowName3))
        A_T_B = makeood(cv2.getTrackbarPos('A_T_B', WindowName3))
        A_T_W = makeood(cv2.getTrackbarPos('A_T_W', WindowName3))
        T_V = float(cv2.getTrackbarPos('T_V', WindowName3))
        
        # mendapatkan nilai dari tracbar Color Thresholding
        RGB = cv2.getTrackbarPos("RGBSwitch", WindowName2)

        Ru = cv2.getTrackbarPos('Ru', WindowName2)
        Gu = cv2.getTrackbarPos('Gu', WindowName2)
        Bu = cv2.getTrackbarPos('Bu', WindowName2)

        Rl = cv2.getTrackbarPos('Rl', WindowName2)
        Gl = cv2.getTrackbarPos('Gl', WindowName2)
        Bl = cv2.getTrackbarPos('Bl', WindowName2)

        lower = np.array([Bl, Gl, Rl], dtype=np.uint8)
        upper = np.array([Bu, Gu, Ru], dtype=np.uint8)

        Xtrans = (Xtrans - 100)
        Ytrans = (Ytrans - 100)
        rotationx = -(X - 100) / 20000.0
        rotationy = (Y - 100) / 20000.0
        rotationz = -(Z - 100) / 100.0
        dist = 1.0 - (W - 100) / 200.0
        stretchX = 1.0 + (D - 100) / -200.0
        w = np.size(image, 1)
        h = np.size(image, 0)
        panX = 0
        panY = 0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # membuat gambar menjadi Gray conversion 

        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # mencoba pelatihan KNN
        if blnKNNTrainingSuccessful == False:  # jika pelatihan KNN tidak berhasil
            print("\nerror: KNN traning was not successful\n")  # show error message
            return
        imaged = imutils.translate(image, Xtrans, Ytrans)
        # menerapkan transformasi
        M = imutils.getTransform(w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist)
        imgOriginalScene = cv2.warpPerspective(imaged, M, (w, h), cv2.INTER_CUBIC or cv2.WARP_INVERSE_MAP)

        if (S == 1):
            imgGrayscale = pp.extractValue(imgOriginalScene) # membuat wadah yang akan menampung hasil dari pemprosesan grayscale
            imgMaxContrastGrayscale = pp.maximizeContrast(imgGrayscale) # memaksimalkan kontras pada gambar setelah proses grayscale
            imgMaxContrastGrayscale = np.invert(imgMaxContrastGrayscale) # Menghitung Inversi  bit-wise  dari elemen elemen array-wise
            height, width = imgGrayscale.shape # mengambil nilai pada gambar grayscale
            imgBlurred = np.zeros((height, width, 1), np.uint8)
            imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (G_S_F_H, G_S_F_W), 0) # memasukkan atau mengimplementasikan fungsi  
                                                                                          # library gaussian blur kedalam gambar
            imgOriginalScene = cv2.adaptiveThreshold(imgBlurred, T_V, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, A_T_B, A_T_W) # membuat wadah yang akan menampung fungsi blur,
                                                                                          # adaptive Thresholding gaussian, dan Thresholding
                                                                                          # binnary
        if (RGB == 1):
            imgOriginalScene = cv2.inRange(imgOriginalScene, lower, upper)

        # berikan definisi untuk setiap inisial pada gambar atau jendela
        cv2.putText(imgOriginalScene, "Press 's' to save the value", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, bottomLeftOrigin=False)
        cv2.putText(imgOriginalScene, "Press 'o' to out the value", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, bottomLeftOrigin=False)
        cv2.putText(imgOriginalScene, "Press 'c' to check the result", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, bottomLeftOrigin=False)
        cv2.putText(imgOriginalScene, "Press 'esc' to close all windows", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, bottomLeftOrigin=False)

        cv2.putText(backGround1, "X for rotating the image in x axis", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround1, "Y for rotating the image in y axis", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround1, "Z for rotating the image in z axis", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround1, "ZOOM for Zoom in or Zoom out the image", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround1, "S for streching the image", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    bottomLeftOrigin=False)

        cv2.putText(backGround2, "R,G,B = Red,Green,Blue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    bottomLeftOrigin=False)
        cv2.putText(backGround2, "u,l = Upper and lower", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    bottomLeftOrigin=False)

        cv2.putText(backGround3, "G_S_F_H = GAUSSIAN_SMOOTH_FILTER_SIZE_HEIGHT", (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround3, "G_S_F_H = GAUSSIAN_SMOOTH_FILTER_SIZE_WEIGHT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround3, "A_T_B = ADAPTIVE_THRESH_BLOCK_SIZE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, bottomLeftOrigin=False)
        cv2.putText(backGround3, "A_T_W = ADAPTIVE_THRESH_WEIGHT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                    1, bottomLeftOrigin=False)
        cv2.putText(backGround3, "T_V = THRESHOLD_VALUE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    bottomLeftOrigin=False)

        # menampilkan pada window
        cv2.imshow("image", imgOriginalScene)
        cv2.imshow(WindowName1, backGround1)
        cv2.imshow(WindowName2, backGround2)
        cv2.imshow(WindowName3, backGround3)

        ch = cv2.waitKey(5)

        # chomand switch
        if ch == ord('c'):  # press c to check the result of processing
            Main.searching(imgOriginalScene, True)
            cv2.imshow("check", imgOriginalScene)
            cv2.waitKey(0)
            return

        if S == 1 and ch == ord('p'):  # press c to check the result of processing
            imgOriginalScene = np.invert(imgOriginalScene)
            cv2.imwrite("calib.png", imgOriginalScene)
            cv2.imshow("calib", imgOriginalScene)
            return

        if ch == ord('o'):  # press o to see the value
            print("CAL_VAL =")
            print(w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H, A_T_B, A_T_W,
                  T_V, Xtrans, Ytrans)

        if ch == ord('s'):  # press s to save the value
            CAL_VAL = np.array([[w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H,
                                 A_T_B, A_T_W, T_V, Xtrans, Ytrans]])
            np.savetxt('calibrated_value.txt', CAL_VAL)
            print(w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H, A_T_B, A_T_W,
                  T_V, Xtrans, Ytrans)
            print("Value saved !")

        if ch == 27:  # press esc for exit the calibration
            break

    cv2.destroyAllWindows()
    return

# membuat fungsi make
def makeood(value):
    if (value % 2 == 0):
        value = value - 1
    if (value < 3):
        value = 3
    return value
