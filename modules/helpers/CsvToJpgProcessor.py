import pandas as pd
from skimage import io, img_as_ubyte

for i in range(1, 4):
    csvData = pd.read_csv('../data/mask' + str(i) + '.csv', sep=';', header=None)
    img = csvData.values
    img = img_as_ubyte(img)

    xShape, yShape = img.shape
    img = img[50:(xShape-50), 50:(yShape-50)]

    io.imsave('../data/mask' + str(i) + '.jpg', img)