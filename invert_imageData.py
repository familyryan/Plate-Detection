# GenData.py

# memasukkan library argparse kedalam project
import argparse
# memasukkan library os kedalam project
import os


# memasukkan library math kedalam project
import cv2
# memasukkan library numpy kedalam project
import numpy as np


# module level variables ##########################################################################


###################################################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--image_train",
                    help="path for the images that you're going to invert")
    args = vars(ap.parse_args())
    if args.get("image", True):
        # membaca dalam gambar nomor pelatihan
        imgTrainingNumbers = cv2.imread(args["image_train"])
        if imgTrainingNumbers is None:
            # cetak pesan kesalahan untuk std out
            print("error: image not read from file \n\n")
            # jeda sehingga pengguna dapat melihat pesan kesalahan
            os.system("pause")
            return
    else:
        print("Please add -d or --image_train argument")

    # mendapatkan gambar grayscale
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

    # filter gambar dari skala abu-abu ke hitam dan putih
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                      0,  # membuat piksel yang melewati threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      # gunakan gaussian daripada rata-rata, tampaknya memberikan hasil yang lebih baik
                                      cv2.THRESH_BINARY_INV,
                                      # invert sehingga latar depan akan putih, latar belakang akan hitam
                                      11,  # ukuran lingkungan piksel yang digunakan untuk menghitung nilai ambang batas
                                      2)  # konstanta dikurangi dari rata-rata atau rata-rata tertimbang

    imgTrainingNumbers = np.invert(imgTrainingNumbers)
    cv2.imwrite("invert_" + args["image_train"], imgTrainingNumbers)
    cv2.imwrite("imgThresh_" + args["image_train"], imgThresh)

    return


###################################################################################################
if __name__ == "__main__":
    main()
# end if
