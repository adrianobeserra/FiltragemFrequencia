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
    M = np.exp(-2j * np.pi * k * n / N)
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

def DFT_main(img):
    height, width = img.shape[0], img.shape[1]
    P,Q = 2*width, 2*height

    #Gera uma Imagem [2M, 2N] completada com zeros. Passo 1
    img_preenchida = np.zeros((Q, P), dtype=np.double)

    #Gera uma imagem preenchida contendo a imagem original. Passo 2
    for y in range(height-1):
        for x in range(width-1):
            img_preenchida[y,x] = img[y,x]

    #Centralizar a transformada multiplicando a imagem por (-1)**(y+x). Passo 3
    for y in range(Q-1):
        for x in range(P-1):
            img_preenchida[y,x] = img_preenchida[y,x]*(-1)**(y+x)

    #Calculando a transformada discreta de fourier. Passo 4
    for y in range(Q-1):
        row = img_preenchida[y,: ]
        row = DFT(row)
        img_preenchida[y,: ] = row.real


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
process_image("2.jpg")