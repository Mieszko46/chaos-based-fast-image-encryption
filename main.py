import struct
import sys
import time
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt

CONST_N = 8
CONST_EPSILON = 0.05
CONST_B = 0.4999
CONST_TIME_ITER = 100


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
    if x <= CONST_B:
        return x / CONST_B
    else:
        return (1.0 - x) / (1.0 - CONST_B)


def generateNCML(x):
    for n in range(CONST_TIME_ITER):
        x_new = np.copy(x)
        for i in range(CONST_N):
            x_new[i] = ((1.0 - CONST_EPSILON) * calculateTentMap(x[i]) +
                        CONST_EPSILON * calculateTentMap(x[(i + 1) % CONST_N]))
        x = x_new
    return x


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
    for i in range(len(S)):
        R[i] = ((sbox[S[i]] ^ (sbox[(S[i] + 1) % 64])) + 
                (sbox[(S[i] + 2) % 64] ^ sbox[(S[i] + 3) % 64])) % 256
    return R


def generateKey():
    return np.random.randint(0, 10, size=16)

def calculate_kl(key, num):
    Ki = key[0]
    for K_next in key[1:]:
        Ki = Ki ^ K_next
    return np.floor(Ki * (num / 256))

def calculate_r(key, totalBits):
    Ki = key[0]
    for K_next in key[1:]:
        Ki = Ki + K_next
    return (Ki % totalBits)


def generatePseudoRandomNumbers(iterate, sboxArray, x_init):
    A = np.zeros(16, dtype=int)

    for loop in range(iterate):
        x_array = generateNCML(x_init)
        for i in range(len(x_array)):
            A[i * 2] = int(binary(x_array[i])[11:18], 2)
            A[i * 2 + 1] = int(binary(x_array[i])[19:26], 2)
        x_init = x_array

    X0 = x_array[0]
    S = generateArrayS(A)
    R = generateArrayR(S, sboxArray)
    return R, X0


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    First dim is the index of block, second is row and third column
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return np.array((arr.reshape(h // nrows, nrows, -1, ncols)
                     .swapaxes(1, 2)
                     .reshape(-1, nrows, ncols)))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h // nrows, -1, nrows, ncols)
            .swapaxes(1, 2)
            .reshape(h, w))


def decimalLSB3(x):
    bin = format(int(x), '08b')
    return int(bin[-3:], 2)


def leftCyclicShift(shift, sequence):
    if sequence == 0:
        return sequence
    sequence_bits = sequence.bit_length()
    shift = shift % sequence_bits
    shifted_num = ((sequence << shift) | (sequence >> (sequence_bits - shift))) & ((1 << sequence_bits) - 1)
    return shifted_num


def rightCyclicShift(shift, sequence):
    if sequence == 0:
        return sequence
    sequence_bits = sequence.bit_length()
    shift = shift % sequence_bits
    shifted_num = ((sequence >> shift) | (sequence << (sequence_bits - shift))) & ((1 << sequence_bits) - 1)
    return shifted_num


def calculateKNew(x0, num):
    return np.floor(x0 * num)


def isNotOccupied(blockValue):
    if blockValue == -1:
        return True
    return False


def switchBlocks(block, kNew, kL, encryptedBlocks, num):
    kNew = int(kNew)
    if kNew != kL and isNotOccupied(encryptedBlocks[kNew][0][0]):
        encryptedBlocks[kNew, :, :] = block
    else:
        while True:
            kNew = (kNew + 1) % num
            if isNotOccupied(encryptedBlocks[kNew][0][0]):
                encryptedBlocks[kNew, :, :] = block
                break
            else:
                continue


def switchBlocksDecrypt(block, kNew, kL, encryptedBlocks, num):
    kNew = int(kNew)
    if kNew == kL and not(isNotOccupied(encryptedBlocks[kNew][0][0])):
        encryptedBlocks[kNew, :, :] = block
    else:
        while True:
            kNew = (kNew + 1) % num
            if isNotOccupied(encryptedBlocks[kNew][0][0]):
                encryptedBlocks[kNew, :, :] = block
                break
            else:
                continue


def encryptImage(sboxArray):
    img = Image.open("./images/castle.png", 'r')
    height, width = img.size
    totalBits = width * height
    mode = img.mode

    encryptedImage = np.zeros((512, 512, 3)) - 1
    encryptedR = encryptedImage[:, :, 0]
    encryptedG = encryptedImage[:, :, 1]
    encryptedB = encryptedImage[:, :, 2]
    encryptedR = blockshaped(encryptedR, 8, 8)
    encryptedG = blockshaped(encryptedG, 8, 8)
    encryptedB = blockshaped(encryptedB, 8, 8)
    encryptedImage = np.stack((encryptedR, encryptedG, encryptedB), axis=3)

    # Step 1
    K = generateKey()
    num = int(totalBits / 64)
    kl = calculate_kl(K, num)
    r = calculate_r(K, totalBits)

    # Step 2
    x0_init = np.zeros(CONST_N)
    for i in range(CONST_N):
        x0_init[i] = (K[i] + 0.1) / 256
    x_init_random = generateNCML(x0_init)

    # Step 3 
    I = np.array(img)
    blocksR = blockshaped(I[:, :, 0], 8, 8)
    blocksG = blockshaped(I[:, :, 1], 8, 8)
    blocksB = blockshaped(I[:, :, 2], 8, 8)
    B = np.stack((blocksR, blocksG, blocksB), axis=3)

    # Step 4 (i)
    randomNumbers, X0 = generatePseudoRandomNumbers(1, sboxArray, x_init_random)
    randomNumbers = np.reshape(randomNumbers, (8, 8))

    # (ii - v)
    C = np.zeros((num, 8, 8, 3), dtype=int)
    G = len(np.unique(I))
    for k in range(num):
        if k == 0:
            for j in range(8):
                C[k][0][j] = K[j + 8]
        for i in range(8):
            for j in range(8):
                for c in range(3):
                    x = ((int(B[k][i][j][c]) ^ int(randomNumbers[i][j])) + C[k][i - 1][j][c]) % G
                    y = decimalLSB3(x) * (int(C[k][i - 1][(j - 1) % 8][c]) ^ int(randomNumbers[i][j]))
                    C[k][i][j][c] = leftCyclicShift(int(x), int(y))
        for c in range(3):
            switchBlocks(C[k, :, :, c], calculateKNew(X0, num), kl, encryptedImage[:, :, :, c], num)
            if k == num - 1:
                encryptedImage[int(kl), :, :, c] = C[int(k), :, :, c]
        # print(k)

    Red = unblockshaped(encryptedImage[:, :, :, 0], height, width)
    Green = unblockshaped(encryptedImage[:, :, :, 1], height, width)
    Blue = unblockshaped(encryptedImage[:, :, :, 2], height, width)

    encryptedImageToSave = np.stack((Red, Green, Blue), axis=2)
    encryptedImageToSave = Image.fromarray(encryptedImageToSave.astype('uint8'), mode)
    encryptedImageToSave.save("./images/result_enc.png")
    return K, kl, r, x_init_random, encryptedImage, C


def decryptImage(sboxArray, K, kl, r, x_init_random, encryptedImage, C):
    img = Image.open("./images/result_enc.png", 'r')
    height, width = img.size
    totalBits = width * height
    mode = img.mode
    num = int(totalBits / 64)

    decryptedImage = np.zeros((512, 512, 3)) - 1
    encryptedR = decryptedImage[:, :, 0]
    encryptedG = decryptedImage[:, :, 1]
    encryptedB = decryptedImage[:, :, 2]
    encryptedR = blockshaped(encryptedR, 8, 8)
    encryptedG = blockshaped(encryptedG, 8, 8)
    encryptedB = blockshaped(encryptedB, 8, 8)
    decryptedImage = np.stack((encryptedR, encryptedG, encryptedB), axis=3)

    # Step 4 (ii)
    randomNumbers, X0 = generatePseudoRandomNumbers(1, sboxArray, x_init_random)
    randomNumbers = np.reshape(randomNumbers, (8, 8))

    # Step 4 (iii)
    P = np.zeros((num, 8, 8, 3), dtype=int)
    I = np.array(img)
    G = len(np.unique(I))
    for k in range(num):
        for c in range(3):
            switchBlocksDecrypt(encryptedImage[k, :, :, c], calculateKNew(X0, num), kl, decryptedImage[:, :, :, c], num)
            # if k == num - 1:
            #     decryptedImage[int(kl), :, :, c] = C[int(k), :, :, c]
        if k == 0:
            for j in range(8):
                P[k][0][j] = K[j + 8]
        for i in range(8):
            for j in range(8):
                for c in range(3):
                    x = C[k][i][j][c]
                    y = decimalLSB3(x) * int(C[k][i - 1][(j - 1) % 8][c]) ^ int(randomNumbers[i][j])
                    P[k][i][j][c] = int(randomNumbers[i][j]) ^ (rightCyclicShift(int(x), int(y)) - C[k][i - 1][j][c] + G) % G
        # print(k)

    Red = unblockshaped(P[:, :, :, 0], height, width)
    Green = unblockshaped(P[:, :, :, 1], height, width)
    Blue = unblockshaped(P[:, :, :, 2], height, width)

    decryptedImage = np.stack((Red, Green, Blue), axis=2)
    decryptedImage = Image.fromarray(decryptedImage.astype('uint8'), mode)
    decryptedImage.save("./images/result_dec.png")


def RED(R): return '#%02x%02x%02x'%(R,0,0)
def GREEN(G): return '#%02x%02x%02x'%(0,G,0)
def BLUE(B):return '#%02x%02x%02x'%(0,0,B)


def showHistograms():
    img = Image.open("./images/result_enc.png", 'r')
    data = np.array(list(img.getdata()), dtype=int)

    img2 = Image.open("./images/result_dec.png", 'r')
    data2 = np.array(list(img2.getdata()), dtype=int)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    
    axs[0].hist(data, 
        bins=256, 
        range=(0,256),          
        density=True, 
        stacked=True, 
        color=("#FF3333", "#33FF56", "#335CFF"))

    axs[1].hist(data2, 
        bins=256, 
        range=(0,256),          
        density=True, 
        stacked=True, 
        color=("#FF3333", "#33FF56", "#335CFF"))
    plt.show() 


def main():
    sboxArray = np.zeros(256, dtype=int)
    getSbox(sboxArray, '.\s-blocks\sbox_08x08_20130117_030729__Original.SBX')

    startTime = time.time()
    K, kl, r, x_init, encryptedImage, C = encryptImage(sboxArray)
    endTime = time.time()
    elapsedTime = endTime - startTime
    print("Encoding time: ", elapsedTime, "seconds")

    startTime = time.time()
    decryptImage(sboxArray, K, kl, r, x_init, encryptedImage, C)
    endTime = time.time()
    elapsedTime = endTime - startTime
    print("Decoding time: ", elapsedTime, "seconds")

    showHistograms()

if __name__ == '__main__':
    main()
