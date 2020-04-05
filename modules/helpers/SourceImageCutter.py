from skimage import io

for i in range(1, 4):
    img = io.imread('../../data/input/pred' + str(i) + '.bmp')
    xShape, yShape = img.shape
    img = img[50:(xShape-50), 50:(yShape-50)]
    io.imsave('../../data/input/source' + str(i) + '.jpg', img)