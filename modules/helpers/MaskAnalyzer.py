class MaskAnalyzer:


    maskProcessor = object()

    def __init__(self, maskProcessor):
        self.maskProcessor = maskProcessor
        self.prepareAnalisysImagesForMask()

    def prepareAnalisysImagesForMask(self):
        shape = self.maskProcessor.mask.shape
        for y in range(20, shape[0], 20):
            for x in range(20, shape[1], 20):
                if self.maskProcessor.mask[y][x] == True:
                    continue
                self.maskProcessor.prepareAnalysiInDot(y, x)