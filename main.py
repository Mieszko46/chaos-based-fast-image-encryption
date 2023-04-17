import random
import sys
from PIL import Image
import numpy as np


def getSbox(sboxArray, src):
    try:
        file = open(src, "rb")
    except:
        print("Can't find such file (wrong path)")
    else:
        row = 0
        while True:
            byte = file.read(1)
            file.seek(1,1)
            if not byte:
                break
            
            sboxArray[row] = int.from_bytes(byte, byteorder=sys.byteorder)
            row += 1
        file.close()


def generateInverseSbox(sboxArray):
    inverseSbox = {v: k for k, v in enumerate(sboxArray)}
    return inverseSbox


def encodeImage(sboxArray):

    img = Image.open("./images/lena.png", 'r')    
    height, width = img.size
    totalBits = width * height

    data = np.array(list(img.getdata()), dtype=int)

    # obrazy w odcieniach szarości są jednowymiarową tablicą, nie dwu, dlatego try catch na kanały w obrazie
    try:
        np.array(img).shape[2]

    except:
        for bit in range(totalBits):
            data[bit] = sboxArray[data[bit]]
        data = data.reshape(width, height)
        mode = 'L'
        
    else:
        channels = np.array(img).shape[2]
        for bit in range(totalBits):
            for c in range(channels):
                data[bit][c] = sboxArray[data[bit][c]]
        data = data.reshape(width, height, channels)
        mode = img.mode

    encryptedImage = Image.fromarray(data.astype('uint8'), mode)
    encryptedImage.save("./images/result_enc.png")


def decodeImage(sboxArray):
    img = Image.open("./images/result_enc.png", 'r')
    height, width = img.size
    totalBits = width * height

    data = np.array(list(img.getdata()), dtype=int)
    # getSbox(sboxArray, ".\sblocks\sbox_08x08_20130117_030729___Inverse.SBX")
    # inverseSbox = sboxArray
    inverseSbox = generateInverseSbox(sboxArray)

    try:
        np.array(img).shape[2]

    except:
        for bit in range(totalBits):
            data[bit] = inverseSbox[data[bit]]
        data = data.reshape(width, height)
        mode = 'L'

    else:
        channels = np.array(img).shape[2]
        for bit in range(totalBits):
            for c in range(channels):
                data[bit][c] = inverseSbox[data[bit][c]]
        data = data.reshape(width, height, channels)
        mode = img.mode

    decryptedImage = Image.fromarray(data.astype('uint8'), mode)
    decryptedImage.save("./images/result_dec.png")


def main():
    sboxArray = np.zeros(256, dtype=int)
    getSbox(sboxArray, '.\s-blocks\sbox_08x08_20130117_030729__Original.SBX')
    print(sboxArray)
    encodeImage(sboxArray)
    decodeImage(sboxArray)


if __name__ == '__main__':
    main()