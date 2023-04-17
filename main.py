import random
import struct
import sys
from PIL import Image
import numpy as np
import copy
from datetime import datetime

CONST_N = 8
CONST_EPSILON = 0.05
CONST_B = 0.4999
CONST_X1 = 0.78


def getSbox(sboxArray, src):
    try:
        file = open(src, "rb")
    except:
        print("Can't find such file (wrong path)")
    else:
        row = 0
        while True:
            byte = file.read(1)
            file.seek(1, 1)
            if not byte:
                break

            sboxArray[row] = int.from_bytes(byte, byteorder=sys.byteorder)
            row += 1
        file.close()


def generateInverseSbox(sboxArray):
    inverseSbox = {v: k for k, v in enumerate(sboxArray)}
    return inverseSbox


def calculateTentMap(x):
    if x == 1:
        x = CONST_X1
        return x

    prevx = calculateTentMap(x - 1)
    if prevx <= CONST_B:
        return prevx / CONST_B

    if prevx > CONST_B:
        return (1.0 - prevx) / (1.0 - CONST_B)


def generateNCML(index):
    result = ((1.0 - CONST_EPSILON) * calculateTentMap(index) +
              CONST_EPSILON * calculateTentMap(index + 1.0))
    return result


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


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


def function_F(a, b, c, d):
    return ((a & b) | ((~a) & c) + d) % 256


def function_G(a, b, c, d):
    return ((a & c) | (b & (~c)) + d) % 256


def function_H(a, b, c, d):
    return ((a ^ b ^ c) + d) % 256


def function_I(a, b, c, d):
    return (b ^ (a | (~c)) + d) % 256


def shiftArrayToRight(A):
    A_copy = copy.deepcopy(A)
    A_copy = np.insert(A_copy, 0, A_copy[len(A_copy) - 1])
    A_copy = np.delete(A_copy, len(A_copy) - 1)
    return A_copy


def generateArrayS(A):
    A_copy = copy.deepcopy(A)
    s = []
    for i in range(4):
        temp = np.zeros(16, dtype=int)
        # print(A_copy)
        for i in range(16):
            if i % 4 == 0:
                temp[i] = function_F(A_copy[i], A_copy[i + 1], A_copy[i + 2], A_copy[i + 3])
            elif i % 4 == 1:
                temp[i] = function_G(A_copy[i], A_copy[i + 1], A_copy[i + 2], A_copy[i - 1])
            elif i % 4 == 2:
                temp[i] = function_H(A_copy[i], A_copy[i + 1], A_copy[i - 2], A_copy[i - 1])
            elif i % 4 == 3:
                temp[i] = function_I(A_copy[i], A_copy[i - 3], A_copy[i - 2], A_copy[i - 1])
        s = np.concatenate((s, temp), axis=0)
        A_copy = shiftArrayToRight(A_copy)
    return np.array(s, dtype=int)


def generateArrayR(S, sbox):
    R = np.zeros(64, dtype=int)
    print(sbox.shape)
    print(S.shape)
    for i in range(len(S)):
        R[i] = ((sbox[S[i]] ^ (sbox[S[i] + 1 % 64])) + (sbox[(S[i] + 2) % 64] ^ sbox[(S[i] + 3) % 64])) % 256
    return R


def main():
    sboxArray = np.zeros(256, dtype=int)
    getSbox(sboxArray, '.\s-blocks\sbox_08x08_20130117_030729__Original.SBX')
    # print(sboxArray)

    time = datetime.now()
    x = np.zeros(CONST_N)
    A = np.zeros(16, dtype=int)
    for i in range(1, CONST_N + 1):
        x[i - 1] = generateNCML(i)
        # print(f'x{i}: {x[i - 1]}')
        # print(f'bin x{i}: {binary(x[i - 1])}')
        A[(i - 1) * 2] = int(binary(x[i - 1])[11:18], 2)
        A[(i - 1) * 2 + 1] = int(binary(x[i - 1])[19:26], 2)
    # print(A)
    S = generateArrayS(A)
    R = generateArrayR(S, sboxArray)
    print("Array S: " + str(S))
    print("Array R: " + str(R))
    # encodeImage(sboxArray)
    # decodeImage(sboxArray)


if __name__ == '__main__':
    main()
