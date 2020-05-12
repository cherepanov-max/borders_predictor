from skimage import io, img_as_bool,img_as_ubyte
from .MaskProcessor import MaskProcessor
from .BordersPredictor import predict_borders

class AppManager:
    """   Управление потоком приложения   """


    def __init__(self, input_path, output_path):
        """   Конструктор   """
        # @var input_path string - Путь к входному изображению
        # @var output_path string - Путь к выходному изображению

        self.image = img_as_bool(io.imread(input_path))
        self.output_path = output_path

    def process_image(self):
        """   Обработать изображение   """

        mask = predict_borders(self.image)
        mask = MaskProcessor(mask).mask
        io.imsave(self.output_path, img_as_ubyte(mask))
        return self.output_path