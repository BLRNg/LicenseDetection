import cv2
import pytesseract
from keras.preprocessing import image

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé


from lib_detection import load_model, detect_lp, im2single, fine_tune

# uploaded = files.upload()
# cap  = image.load_img('IMG_1418.jpg', target_size=(400,642))


# # Load model LP detection
# wpod_net_path = "wpod-net_update1.json"
# wpod_net = pd.read_json("/content/MiAI_LP_Detection_2/wpod-net_update1.json")

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
# img_path = "/content/MiAI_LP_Detection_2/test/xemay.jpg"

img_path = "dataset/0403_07084_b.jpg"
# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)
# cv2.imshow("Anh goc",Ivehicle)
# cv2.waitKey(0)


# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)



_, LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
print(lp_type, 'type')
# cv2.imshow("Anh goc",LpImg[0])
# cv2.waitKey(0)
LpImg[0] = LpImg[0][5:int(LpImg[0].shape[0])-5, 10:int(LpImg[0].shape[1])].copy()

y = int(LpImg[0].shape[0] / 2)
print(y)
x = int(LpImg[0].shape[1])
print(x)
imagePart1 = LpImg[0][0:y, 0:x].copy()
# cv2.imshow("Anh goc",imagePart1)
# cv2.waitKey(0)

y = int(LpImg[0].shape[0] / 2)
print(y)
x = int(LpImg[0].shape[1])
print(LpImg[0].shape[0]/LpImg[0].shape[1])
imagePart2=LpImg[0][y:y*2, 0:x].copy()
# cv2.imshow("Anh goc",imagePart2)
# cv2.waitKey(0)

if (len(LpImg)):

    # Chuyen doi anh bien so
    imagePart2 = cv2.convertScaleAbs(imagePart2, alpha=(255.0))

    # Chuyen anh bien so ve gray
    gray = cv2.cvtColor( imagePart2, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Anh bien so sau chuyen xam", gray)
    # cv2.waitKey(0)

    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # cv2.imshow("Anh bien so sau threshold", binary)
    # cv2.waitKey(0)

    # Nhan dien bien so. Cau hinh --psm 7 la de nhan dien 1 line only
    # text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7")
    text = pytesseract.image_to_string(binary, lang="eng",config="--psm 11 --oem 3")
    # cv2.imshow(text, Ivehicle)
    # cv2.waitKey(0)
    print(fine_tune(text))
#     # Viet bien so len anh
#     cv2.putText(Ivehicle,fine_tune(text),(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

#     # Hien thi anh va luu anh ra file output.png
#     cv2.imshow("Anh input", Ivehicle)
#     cv2.imwrite("output.png",Ivehicle)
#     cv2.waitKey()
      # Chuyen doi anh bien so
    imagePart1 = cv2.convertScaleAbs(imagePart1, alpha=(255.0))

    # Chuyen anh bien so ve gray
    gray = cv2.cvtColor( imagePart1, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Anh bien so sau chuyen xam", gray)
    # cv2.waitKey(0)

    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 127, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # cv2.imshow("Anh bien so sau threshold", binary)
    # cv2.waitKey(0)

    # Nhan dien bien so. Cau hinh --psm 7 la de nhan dien 1 line only
    # text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7")
    text = pytesseract.image_to_string(binary, lang="eng",config="--psm 11 --oem 3")
    print(fine_tune(text))
    ###############################
        # Chuyen doi anh bien so
    # LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    #
    # # Chuyen anh bien so ve gray
    # gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)
    #
    # # cv2.imshow("Anh bien so sau chuyen xam", gray)
    # # cv2.waitKey(0)
    #
    # # Ap dung threshold de phan tach so va nen
    # binary = cv2.threshold(gray, 127, 255,
    #                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #
    # # cv2.imshow("Anh bien so sau threshold", binary)
    # # cv2.waitKey(0)
    #
    # # Nhan dien bien so. Cau hinh --psm 7 la de nhan dien 1 line only
    # # text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7")
    # text = pytesseract.image_to_string(binary, lang="eng",config="--psm 11 --oem 3")
    # print(text)