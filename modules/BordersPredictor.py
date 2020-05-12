from skimage import img_as_float
import numpy as np
from keras.models import load_model


class BordersPredictor:
    """ Предсказания границ изображения через нейросеть """

    # Размер окна фильтра
    WINDOW = 50

    # Размер половины окна фильтра
    HALF = 25

    def __init__(self, image):
        """   Конструктор
        :param image numpy array - изображение
        """

        self.model = load_model('neural_network_model.h5')
        self.image = self.process_input_image(image)


    def process_input_image(self, image):
        """   Обработка входного изображения под формат, подходящий для рпедсказания
        :param image numpy array - изображение
        """
        work = img_as_float(image)
        image = np.zeros((work.shape[0], work.shape[1], 1))
        for y in range(work.shape[0]):
            for x in range(work.shape[1]):
                image[y][x] = [work[y][x]]
        return image


    def predict(self):
        """   Предсказать границы
        :param image numpy array - изображение
        """
        mask = np.zeros(self.image.shape)
        for Y in (range(mask.shape[0] - self.WINDOW)):
            for X in (range(mask.shape[1] - self.WINDOW)):
                y = int(Y + self.HALF)
                x = int(X + self.HALF)
                inputData = np.array([self.image[(y - self.HALF):(y + self.HALF), (x - self.HALF):(x + self.HALF)]])
                pred = round(self.model.predict(inputData)[0][0])
                mask[y][x] = pred
        return mask


def predict_borders(image):
    """   Предсказать границы для изображения
    :param image numpy array - изображение
    """

    predictor = BordersPredictor(image)
    return predictor.predict()

