import cv2
import numpy as np
import time
import sys
import os

def DFT(x):
    """Calcula a transformada discreta de Fourier 2D"""
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, x)

def DFT_inversa(x):
    """Calcula a transformada discreta de Fourier 2D"""
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = 1/N*(np.exp(-2j * np.pi * k * n / N))
    return np.dot(M, x)

def getFiltro(tamanhoJanela):
    filtro = np.zeros((tamanhoJanela, tamanhoJanela), np.double)
    return filtro

def DFT_main(img):
    height, width = img.shape[0], img.shape[1]
    P,Q = 2*width, 2*height
    #imgDest = np.zeros((height, width), dtype=complex)
    imgDest = np.zeros((height, width), dtype=np.double)
    img_preenchida = np.zeros((Q, P), dtype=np.double)
    filtro = getFiltro(3)
    filter_width, filter_height = filtro.shape[0], filtro.shape[1]

    filtro[0,0] = 1
    filtro[0,1] = 1
    filtro[0,2] = 1
    filtro[1,0] = 1
    filtro[1,1] = -8
    filtro[1,2] = 1
    filtro[2,0] = 1
    filtro[2,1] = 1
    filtro[2,2] = 1

    for y in range(height-1):
        for x in range(width-1):
            img_preenchida[y,x] = img[y,x]

    for y in range(Q-1):
        for x in range(P-1):
            img_preenchida[y,x] = img_preenchida[y,x]*(-1)**(y+x)

    for y in range(Q):
        row = img_preenchida[y,: ]
        row = DFT(row)
        img_preenchida[y,: ] = row.real

    for y in range(height):
        for x in range(width):
            sum_pixels = 0
            for filterY in range(int(-(filter_height / 2)), filter_height - 1):
                for filterX in range(int(-(filter_width / 2)), filter_width - 1):
                    pixel_y = y - filterY
                    pixel_x = x - filterX
                    pixel = img_preenchida[filterY, filterX]

                    #Defini a ação a ser realizada quando o pixel atual fizer parte das bordas da imagem.
                    if (pixel_y >= 0) and (pixel_y < height) and (pixel_x >= 0) and (pixel_x < width):
                        pixel = img_preenchida[pixel_y, pixel_x]

                    img_preenchida[filterY, filterX] = img_preenchida[filterY, filterX] * filtro[filterY, filterX]

    for y in range(Q):
        row = img_preenchida[y,: ]
        row = DFT_inversa(row)
        img_preenchida[y,: ] = row.real

    for y in range(Q-1):
        for x in range(P-1):
            img_preenchida[y,x] = img_preenchida[y,x]*(-1)**(y+x)

    return img_preenchida

def process_image(imgName):
    start_time = time.time()
    desImgName = imgName
    imgName = sys.path[0] + '\\' + imgName
    processedFolder = sys.path[0] + '\\' + 'processed'

    if not os.path.exists(processedFolder):
        os.makedirs(processedFolder)

    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    print("Processing image '{0}'...".format(imgName))
    imgDest = DFT_main(img)
    cv2.imwrite(processedFolder + "\\" + desImgName, imgDest)
    elapsed_time = time.time() - start_time
    print("Done.")
    print("Done! Elapsed Time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

    ''' Caso queira exibir a imagem na tela 
    cv2.imshow(imgName, imgDest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

'''Programa Principal'''
process_image("Agucar_(1).jpg")