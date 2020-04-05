from skimage import io, img_as_ubyte, img_as_bool
from skimage.morphology import square, binary_closing, remove_small_objects

for i in range(1, 4):
    img = img_as_bool(io.imread('../../data/input/mask' + str(i) + '.jpg'))
    img = binary_closing(img, square(25))
    img = remove_small_objects(img)
    img = img_as_ubyte(img)
    io.imsave('../../data/input/processed_mask' + str(i) + '.jpg', img)