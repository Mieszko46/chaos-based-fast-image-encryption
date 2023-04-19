import struct
import sys
from PIL import Image
import numpy as np
import copy

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
    # print(sbox.shape)
    # print(S.shape)
    for i in range(len(S)):
        R[i] = ((sbox[S[i]] ^ (sbox[(S[i] + 1) % 64])) + (sbox[(S[i] + 2) % 64] ^ sbox[(S[i] + 3) % 64])) % 256
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


def generatePseudoRandomNumbers(iterate, sboxArray):
    # time = datetime.now()
    x_init = np.random.rand(CONST_N)
    A = np.zeros(16, dtype=int)

    for loop in range(iterate):
        x_array = generateNCML(x_init)
        for i in range(len(x_array)):
            A[i * 2] = int(binary(x_array[i])[11:18], 2)
            A[i * 2 + 1] = int(binary(x_array[i])[19:26], 2)
        x_init = x_array

    X0 = x_array[0]
    # print("Array A: ", A)
    S = generateArrayS(A)
    R = generateArrayR(S, sboxArray)
    # print("Array S: " + str(S))
    # print("Array R: " + str(R))
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


def calculateKNew(x0, num):
    return np.floor(x0 * num)


# Assumes that block full of -1 is unoccupied
def isNotOccupied(array, kNew):
    arr = copy.deepcopy(array)
    arr = np.array(arr, dtype=int)
    temp = np.zeros((arr.shape[1], arr.shape[2]), dtype=int) - 1
    checkedBlock = np.array(arr[int(kNew), :, :], dtype=int)
    if (checkedBlock == temp).all():
        # print(True)
        return True
    else:
        # print(False)
        return False


def switchBlocks(block, kNew, kL, encryptedBlocks, num):
    # print("knew: ", kNew, kL)
    kNew = int(kNew)
    if kNew != kL and isNotOccupied(encryptedBlocks, kNew):
        encryptedBlocks[kNew, :, :] = block
    else:
        while True:
            kNew = (kNew + 1) % num
            if isNotOccupied(encryptedBlocks, kNew):
                encryptedBlocks[kNew, :, :] = block
                break
            else:
                continue


def encryptImage(sboxArray):
    img = Image.open("./images/lena.png", 'r')
    height, width = img.size
    totalBits = width * height

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
    print("Key K: ", K)
    num = int(totalBits / 64)
    kl = calculate_kl(K, num)
    print("kl: ", kl)
    r = calculate_r(K, totalBits)
    print("r: ", r)

    # Step 2
    x0_init = np.zeros(CONST_N)
    for i in range(CONST_N):
        x0_init[i] = (K[i] + 0.1) / 256
    print("init array: ", x0_init)
    x_array = generateNCML(x0_init)
    print("N0 NCML: ", x_array)

    # b = img2.reshape((8, 8, img2.shape[0] * img2.shape[1]))
    # Loop for encoding each block, it will be needed
    # for i in range(num):
    # Step 3  what the fck is going on in this step
    I = np.array(img)
    blocksR = blockshaped(I[:, :, 0], 8, 8)
    blocksG = blockshaped(I[:, :, 1], 8, 8)
    blocksB = blockshaped(I[:, :, 2], 8, 8)
    B = np.stack((blocksR, blocksG, blocksB), axis=3)
    # print(B.shape)
    # print(I.shape)

    # pseudo Step 3 (remove hen Step 3 will be finished)
    # B = np.zeros((num, 8, 8))
    # for i in range(num):
    #     B[i] = np.random.randint(0, 255, size=(8, 8))
    # print(B.shape)

    # Step 4 (i)
    randomNumbers, X0 = generatePseudoRandomNumbers(1, sboxArray)
    randomNumbers = np.reshape(randomNumbers, (8, 8))

    # print(B)

    # (ii)
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

    print(encryptedImage)
    # print(C)

    # encryptedImage = Image.fromarray(data.astype('uint8'), mode)
    # encryptedImage.save("./images/result_enc.png")


def decryptImage(sboxArray):
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
    # print(sboxArray)
    encryptImage(sboxArray)
    # decryptImage(sboxArray)


if __name__ == '__main__':
    main()
