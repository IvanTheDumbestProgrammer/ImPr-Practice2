import cv2
import matplotlib.pyplot as plt
import numpy as np

def histogramm(frame):
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))
    plt.title('Color Histogram')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')

    for i, col in enumerate(colors):
        histogram, bin_edges = np.histogram(
            frame[:, :, i], bins=256, range=(0, 255)
        )
        plt.plot(bin_edges[0:-1], histogram, color=col, label=f'{col.upper()} channel')

    plt.legend()
    plt.show()



while True:
    print('Работать с изображением или с видео? 1/2')
    case1 = int(input())
    match case1:
        case 1:
            print('''Что выполнять?:
             1.Составить гистограмму изображения
             2.Фильтр Кэнни
             3.Фильтр Кэнии для размытого изображения
             4.Обнаружить границы с помощью вычетания''')

            case2 = int(input())
            img = cv2.imread("meow.jpg")
            match case2:
                case 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    histogramm(img)
                    cv2.imshow('a', img)
                    cv2.waitKey(0)
                case 2:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    img = cv2.Canny(img, 40, 100)
                    cv2.imshow('Canny', img)
                    cv2.waitKey(0)
                case 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    blur_img = cv2.GaussianBlur(img, (5, 5), 2)
                    edges = cv2.Canny(blur_img, 50, 100)
                    cv2.imshow('Gauss-Canny', edges)
                    cv2.waitKey(0)
                case 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    blur_img = cv2.GaussianBlur(img, (7,7), 2)
                    img_diff = cv2.absdiff(img, blur_img)
                    cv2.imshow('Diff', img_diff)
                    cv2.waitKey(0)
                case _:
                    break
        case 2:
            print('''Что выполнять?:
             1.Составить гистограмму изображения
             2.Фильтр Кэнни
             3.Фильтр Кэнии для размытого изображения
             4.Обнаружить границы с помощью вычетания''')
            case2 = int(input())
            match case2:
                case 1:
                    cap = cv2.VideoCapture(0)

                    while True:
                        _, img = cap.read()
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        img = cv2.blur(img, (3, 3))

                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                        cv2.imshow('window', hsv)
                        histogramm(hsv)

                        if cv2.waitKey(1) == 27:
                            cv2.destroyAllWindows()
                            break
                case 2:
                    cap = cv2.VideoCapture(0)

                    while True:
                        _, img = cap.read()

                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        edges = cv2.Canny(hsv, 100, 150)
                        cv2.imshow('window', edges)

                        if cv2.waitKey(1) == 27:
                            cv2.destroyAllWindows()
                            break
                case 3:
                    cap = cv2.VideoCapture(0)

                    while True:
                        _, img = cap.read()
                        img = cv2.blur(img, (7, 7))

                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        edges = cv2.Canny(hsv, 100, 150)
                        cv2.imshow('window', edges)

                        if cv2.waitKey(1) == 27:
                            cv2.destroyAllWindows()
                            break
                case 4:
                    cap = cv2.VideoCapture(0)

                    while True:
                        _, img = cap.read()
                        img = cv2.GaussianBlur(img, (5, 5), 2)

                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        blur = cv2.GaussianBlur(hsv, (7, 7), 2)

                        edges = cv2.absdiff(hsv, blur)
                        cv2.imshow('window', edges)

                        if cv2.waitKey(1) == 27:
                            cv2.destroyAllWindows()
                            break
                case _:
                    break
        case _:
            break