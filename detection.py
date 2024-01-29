import cv2
import pytesseract
from lib_detection import load_model, detect_lp, im2single, fine_tune

def main(string_from_csharp):
    print(f"Received string from C#: {string_from_csharp}")
    img_path = "C:/Users/trant/PycharmProjects/LicenseDetection/test_image/0129_06028_b.jpg"
    # Load model LP detection
    wpod_net_path = "C:/Users/trant/PycharmProjects/LicenseDetection/wpod-net_update1.json"
    wpod_net = load_model(wpod_net_path)

    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)


    # Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
    Dmax = 608
    Dmin = 288

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _, LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)



    if (len(LpImg)):
        if (lp_type == 2):

            y = int(LpImg[0].shape[0] / 2)
            x = int(LpImg[0].shape[1])
            imagePart1 = LpImg[0][10:y, 10:x - 0].copy()

            y = int(LpImg[0].shape[0] / 2)
            x = int(LpImg[0].shape[1])
            imagePart2 = LpImg[0][y - 35:y * 2, 15:x - 30].copy()

            # Chuyen doi anh bien so
            imagePart1 = cv2.convertScaleAbs(imagePart1, alpha=(255.0))

            # Chuyen anh bien so ve gray
            gray = cv2.cvtColor( imagePart1, cv2.COLOR_BGR2GRAY)

            # Ap dung threshold de phan tach so va nen
            binary = cv2.threshold(gray, 127, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            text = pytesseract.image_to_string(binary, lang="eng",config="--psm 11 --oem 3")

              # Chuyen doi anh bien so
            imagePart2 = cv2.convertScaleAbs(imagePart2, alpha=(255.0))

            # Chuyen anh bien so ve gray
            gray = cv2.cvtColor( imagePart2, cv2.COLOR_BGR2GRAY)

            # Ap dung threshold de phan tach so va nen
            binary = cv2.threshold(gray, 127, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            text += pytesseract.image_to_string(binary, lang="eng",config="--psm 11 --oem 3")
            print(fine_tune(text))
        else:
            # Chuyen doi anh bien so
            LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
            cv2.imshow('text', LpImg[0])
            cv2.waitKey(0)
            # Chuyen anh bien so ve gray
            gray = cv2.cvtColor(LpImg[0], cv2.COLOR_BGR2GRAY)

            # Ap dung threshold de phan tach so va nen
            binary = cv2.threshold(gray, 127, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            text = pytesseract.image_to_string(binary, lang="eng", config="--psm 11 --oem 3")
            print(fine_tune(text))
        ###############################
        # Viet bien so len anh
        # cv2.putText(Ivehicle, fine_tune(text), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)
        # cv2.imshow(text, Ivehicle)
        # cv2.waitKey(0)

if __name__ == "__main__":
    import sys

    main(sys.argv[1])
