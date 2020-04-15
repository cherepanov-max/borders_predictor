import numpy as np
from skimage import io, img_as_bool, img_as_ubyte
from skimage.morphology import square, binary_closing, binary_dilation, remove_small_objects
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from copy import deepcopy
from skimage.morphology import square, binary_closing, remove_small_objects


class MaskProcessor:
    """   Обработака маски границ камней   """

    # @var integer Шаг для расчитывания массива градусов
    DEGREE_STEP = 1

    # @var integer
    CLOSURE_RATIO = 0.5

    # @var dict Пределы вариации для замыкания границ по проходам
    VARIATION_LIMITS = {
        1:0.45,
        2:0.3
    }

    # @var numpy array (2d) Маска изображения
    mask = []

    # @var int номер прохода замыкания границ
    closing_number = 1

    # @var numpy array (1d) Градусы, по которым будет просчитываться угол
    degreeArray = np.array([])

    # @var int Угол в градусах для поска обратного сильного изменения
    max_steps_to_closing = None

    def __init__(self, maskPath):
        """   Конструктор   """
        # @var maskPath string - Путь к изображению маски
        self.mask = img_as_bool(io.imread(maskPath))
        self.prepare_degree_array()
        self.max_steps_to_closing = int(len(self.degreeArray) / 6)

        self.draw_mask()
        self.mask_morphology_close()
        self.process_mask()
        self.closing_number = 2
        self.process_mask()
        self.draw_mask()

    def draw_mask(self):
        """   Отобразить изображение маски   """
        fig, ax = plt.subplots()
        plt.imshow(self.mask)
        plt.show()
        plt.close('all')

    def process_mask(self):
        """   Обработать маску   """
        shape = self.mask.shape
        for y in range(20, shape[0], 20):
            for x in range(20, shape[1], 20):
                if self.mask[y][x] == True:
                    continue
                print(y, x)
                slice = self.prepareSliceForDot(y, x)
                lengthsArray = self.convertSliceDotsToLenths(y, x, slice)
                if self.calculateVariationForLengthsArray(lengthsArray) > self.VARIATION_LIMITS[self.closing_number]:
                    continue
                derivativeArray = self.prepareDerivativeArray(lengthsArray)
                new_lengths_array, new_derivative_array = self.process_dot_border(y, x, slice, lengthsArray, derivativeArray)
                self.drawBorderInMask(y, x, new_lengths_array, slice)
        self.mask_morphology_close()

    def prepare_degree_array(self):
        """   Подготовить градусы, по которым будет просчитываться угол   """
        degree = 0
        while degree < 360:
            self.degreeArray = np.append(self.degreeArray, degree)
            degree += self.DEGREE_STEP

    def move_to_border(self, degree, step):
        """   Определить координаты по градусу и длине шага   """
        # @var degree int - градус
        # @var step int - номер щага
        # @return int - ордината

        rad = degree * math.pi / 180
        if degree == 0:
            x = step
            y = 0
        elif degree == 90:
            y = step
            x = 0
        elif degree == 180:
            x = -step
            y = 0
        elif degree == 270:
            y = -step
            x = 0
        elif degree > 90 and degree < 270:
            if degree > 135 and degree < 225:
                x = -step
                y = math.tan(rad) * x
            else:
                if degree <= 135:
                    y = step
                    x = y /  math.tan(rad)
                else:
                    y = -step
                    x = y /  math.tan(rad)
        else:
            if degree < 45 or degree > 315:
                x = step
                y = math.tan(rad) * x
            else:
                if degree > 270:
                    y = -step
                    x = y / math.tan(rad)
                else:
                    y = step
                    x = y / math.tan(rad)
        return y, x

    def prepareSliceForDot(self, dotY, dotX):
        """   Построить срез прострелов маски в определенной точке   """
        # @var dotY int - координата У
        # @var dotX int -  координата X
        # @return np.array массив точек прострелов для градусов

        limitsY = (0, self.mask.shape[0] - 1)
        limitsX = (0, self.mask.shape[1] - 1)
        slice = np.empty((0, 2), dtype=int)
        for degree in self.degreeArray:
            step = 1
            while True:
                deltaY, deltaX  = self.move_to_border(degree, step)
                y = int(dotY + deltaY)
                x = int(dotX + deltaX)
                if x >= limitsX[1] or\
                    x <= limitsX[0] or\
                    y >= limitsY[1] or\
                    y <= limitsY[0] or\
                    self.mask[y][x] == True:
                    break
                else:
                    step += 1
            slice = np.concatenate((slice, np.array([[y, x]])))
        return slice

    def convertSliceDotsToLenths(self, dotY, dotX, slice):
        """   Конвертировать срез прострелов маски в определенной точке в массив длин  """
        # @var dotY int - координата У
        # @var dotX int -  координата X
        # @var np.array массив точек прострелов для градусов
        # @return np.array массив длин до границ по градусам

        lengthsArray = np.array([])
        for dot in slice:
            lengthY = abs(abs(dot[0]) - dotY)
            lengthX = abs(abs(dot[1]) - dotX)
            length = (lengthY**2 + lengthX**2)**0.5
            lengthsArray = np.append(lengthsArray, length)
        return lengthsArray

    def calculateVariationForLengthsArray(self, lenthsArray):
        """   Конвертировать срез прострелов маски в определенной точке в массив длин  """
        # @var np.array lenthsArray массив длин до границ по градусам
        # @float вариация выборки
        new_lenths_array = deepcopy(lenthsArray)
        new_lenths_array.sort()
        new_lenths_array = new_lenths_array[10:350]
        return new_lenths_array.std() / new_lenths_array.mean()

    def prepareDerivativeArray(self, lengthsArray):
        """   Получить массив производных для массива длин """
        # @var np.array lenthsArray массив длин до границ по градусам
        # @return np.array массив производных для массива длин

        derivativeArray = np.array([])
        lengthOfArray = lengthsArray.shape[0]
        for index in range(lengthOfArray):
            if index + 1 < lengthOfArray:
                nextIndex = index + 1
            else:
                nextIndex = 0
            derivativeArray = np.append(derivativeArray, lengthsArray[nextIndex] - lengthsArray[index])
        return derivativeArray

    def prepareRollingDerivativeArray(self, derivativeArray, rolling_step):
        rolling_derivative = np.array([])
        ext_derivative_array = np.concatenate((derivativeArray[-rolling_step:], derivativeArray))
        for index in range(rolling_step, len(derivativeArray)+rolling_step):
            rolling_derivative = np.append(rolling_derivative, sum(ext_derivative_array[(index-rolling_step):(index)]))
        return rolling_derivative

    def prepareAnalysiInDot(self, y, x):
        TEMP_MASK = 'data/output/temp_mask.jpg'
        TEMP_DIAGRAM = 'data/output/temp_diagram.jpg'

        slice = (self.prepareSliceForDot(y, x))
        lengthsArray = self.convertSliceDotsToLenths(y, x, slice)
        variation = self.calculateVariationForLengthsArray(lengthsArray)
        if variation >= self.VARIATION_LIMIT_2:
            return
        derivativeArray = self.prepareDerivativeArray(lengthsArray)

        X = []
        Y = []
        for dot in slice:
            Y.append(dot[0])
            X.append(dot[1])
        fig, ax = plt.subplots()
        ax.scatter(X, Y, s=1, marker=",")
        ax.scatter([x], [y], s=1, marker=",")
        ax.imshow(self.mask)
        plt.savefig(TEMP_MASK, dpi=300)

        fig, ax = plt.subplots()
        ax.bar(self.degreeArray, lengthsArray)
        ax.plot(self.degreeArray, derivativeArray, color='red')
        plt.savefig(TEMP_DIAGRAM, dpi=300)

        D = io.imread(TEMP_DIAGRAM)
        M = io.imread(TEMP_MASK)
        # print(M.shape, D.shape)
        common = np.concatenate([M, D], axis=1)
        # print(common.shape)
        # fig, ax = plt.subplots()
        # io.imshow(common)
        finalPath = 'data/collection'
        if variation < self.VARIATION_LIMIT_2:
            finalPath = finalPath + '/good/common_' + str(y) + '_' + str(x) + 'var-' + str(round(variation*100, 0)) + '.jpg'
        else:
            finalPath = finalPath + '/bad/common_' + str(y) + '_' + str(x) + 'var-' + str(round(variation*100, 0)) + '.jpg'
        io.imsave(finalPath, common)
        plt.close('all')

    def close_border(self, y, x, lengthsArray, derivativeArray, start, stop):
        end_lengths = 1 # длина массива перед и после интерполируемыми значениями
        degree_array_length = len(self.degreeArray) # длина массива длин

        # инкрементруем стоп. чтобы не цеплять последнее значение
        stop += 1
        if stop == degree_array_length:
            stop = 0

        # индексы перед и после интерполируемыми значениями
        indexes = np.array([])
        for i in range(start - end_lengths, start):
            index = i
            if index < 0:
                index += degree_array_length
            if index > degree_array_length:
                index -= degree_array_length
            indexes = np.append(indexes, index)
        for i in range(stop, stop + end_lengths):
            index = i
            if index < 0:
                index += degree_array_length
            if index > degree_array_length:
                index -= degree_array_length
            indexes = np.append(indexes, index)

        # значения Х для апроксимации
        x_data = [i for i in range(end_lengths)]
        range_length = stop - start

        # длина массива для интерполяции
        if range_length < 0:
            range_length += degree_array_length
        x_data = np.concatenate((x_data, [i for i in range(end_lengths + range_length, range_length + 2*end_lengths)]))

        # значения Y для апроксимации
        y_data = np.array([])
        for i in indexes:
            y_data = np.append(y_data, lengthsArray[int(i)])

        popt, pcov = curve_fit(linear, x_data, y_data)
        a, b = popt

        for numb, i in enumerate(range(start + 1, start + range_length)):
            index = i
            if i >= degree_array_length:
                index -= degree_array_length
            length_value = linear(numb + end_lengths, a, b)
            lengthsArray[int(index)] = round(length_value, 0)
        return lengthsArray, self.prepareDerivativeArray(lengthsArray)

    def process_dot_border(self, y, x, slice, lengthsArray, derivativeArray):
        limit = lengthsArray.mean() * 0.3
        for index, degree in enumerate(self.degreeArray):
            stop = False
            if derivativeArray[index] > limit:
                stop = self.find_stop_by_outburst(index, derivativeArray)
                if stop != False and self.closing_number == 1:
                    print('close border', index, stop)
                    lengthsArray, derivativeArray = self.close_border(y, x, deepcopy(lengthsArray), deepcopy(derivativeArray), index, stop)
                if stop == False and self.closing_number == 2:
                    stop = self.find_stop_by_one_deviation(index, deepcopy(lengthsArray))
                    if stop != False:
                        print('close border', index, stop)
                        lengthsArray, derivativeArray = self.close_border(y, x, deepcopy(lengthsArray), deepcopy(derivativeArray), index, stop)
        return lengthsArray, derivativeArray

    def drawBorderInMask(self, y, x, lengthsArray, slice):
        # new_slice = []
        for i, degree in enumerate(self.degreeArray):
            border_y = y + math.sin(degree * math.pi / 180) * lengthsArray[i]
            border_y = int(round(border_y))
            if border_y > self.mask.shape[0]:
                border_y = self.mask.shape[0]
            elif border_y < 0:
                border_y = 0
            border_x = x + math.cos(degree * math.pi / 180) * lengthsArray[i]
            border_x = int(round(border_x))
            if border_x > self.mask.shape[1]:
                border_x = self.mask.shape[1]
            elif border_x < 0:
                border_x = 0
            self.mask[border_y-1:border_y+1, border_x-1:border_x+1] = True
            # new_slice.append([border_y, border_x])
        # new_slice = np.array(new_slice)
        # odl_slice = np.array(slice)

        # fix, ax = plt.subplots()
        # ax.scatter(new_slice[:, 1], new_slice[:, 0])
        # ax.scatter(odl_slice[:, 1], odl_slice[:, 0])
        #
        # ax.bar(self.degreeArray, lengthsArray)
        # plt.show()
        # exit()

    def find_stop_by_outburst(self, index, derivativeArray):
        stop_index = index + self.max_steps_to_closing # когда останавливаемся
        degrees_total = len(self.degreeArray) # всего точек
        start_length = derivativeArray[index] # стартовая длина
        while index < stop_index:
            if index < degrees_total:
                current_index = index
            else:
                current_index = index - degrees_total
            if start_length*self.CLOSURE_RATIO < -derivativeArray[current_index] < start_length/self.CLOSURE_RATIO:
                return current_index
            index += 1
        return False

    def find_stop_by_one_deviation(self, index, legths_array):
        degree_array_length = len(self.degreeArray)  # длина массива углов
        filter_size = 3

        legths_array = np.roll(legths_array, degree_array_length - index)
        legths_array = np.append(legths_array[-filter_size:], legths_array)

        stop = False
        was_max = False
        previous = legths_array[:filter_size + 1].mean()
        for i in range(filter_size + 1, degree_array_length + filter_size):
            current = legths_array[i - filter_size:i + 1].mean()
            if was_max == False:
                if current < previous:
                    was_max = True
            else:
                if current > previous:
                    stop = i
                    break
            previous = current
        if stop == False:
            return False
        else:
            stop += index
        if stop >= degree_array_length:
            stop -= degree_array_length
        return stop

    def mask_morphology_close(self):
        self.mask = binary_closing(self.mask, square(25))
        self.mask = remove_small_objects(self.mask)

    def test(self):
        self.closing_number = 2
        # y, x = 60, 340
        # slice = self.prepareSliceForDot(y, x)
        # lengthsArray = self.convertSliceDotsToLenths(y, x, slice)
        # derivativeArray = self.prepareDerivativeArray(lengthsArray)
        # new_lengths_array, new_derivative_array = self.process_dot_border(y, x, slice, lengthsArray, derivativeArray)
        # self.drawBorderInMask(y, x, new_lengths_array, slice)
        # fig, ax = plt.subplots()
        # ax.bar(self.degreeArray, lengthsArray)
        # ax.plot(self.degreeArray, derivativeArray, color='red')
        # plt.show()
        # exit()

        fig, ax = plt.subplots()
        plt.imshow(self.mask)
        plt.show()
        shape = self.mask.shape
        for y in range(20, shape[0], 20):
            for x in range(20, shape[1], 20):
                if self.mask[y][x] == True:
                    continue
                print(y, x)
                slice = self.prepareSliceForDot(y, x)
                lengthsArray = self.convertSliceDotsToLenths(y, x, slice)
                if self.calculateVariationForLengthsArray(lengthsArray) > self.VARIATION_LIMIT_2:
                    continue
                derivativeArray = self.prepareDerivativeArray(lengthsArray)
                new_lengths_array, new_derivative_array = self.process_dot_border(y, x, slice, lengthsArray, derivativeArray)
                self.drawBorderInMask(y, x, new_lengths_array, slice)
        fig, ax = plt.subplots()
        plt.imshow(self.mask)
        plt.show()
        self.mask_morphology_close()
        fig, ax = plt.subplots()
        plt.imshow(self.mask)
        plt.show()
        # io.imsave('data/output/mask_with_borders_1.jpg', img_as_ubyte(self.mask))
        # plt.close('all')
    def test9(self):
        fig, ax = plt.subplots()
        plt.imshow(self.mask)
        plt.show()
        shape = self.mask.shape
        for y in range(20, shape[0], 20):
            for x in range(20, shape[1], 20):
                if self.mask[y][x] == True:
                    continue
                print(y, x)
                slice = self.prepareSliceForDot(y, x)
                lengthsArray = self.convertSliceDotsToLenths(y, x, slice)
                if self.calculateVariationForLengthsArray(lengthsArray) > self.VARIATION_LIMIT_1:
                    continue
                derivativeArray = self.prepareDerivativeArray(lengthsArray)
                new_lengths_array, new_derivative_array = self.process_dot_border(y, x, slice, lengthsArray, derivativeArray)
                self.drawBorderInMask(y, x, new_lengths_array, slice)
        fig, ax = plt.subplots()
        plt.imshow(self.mask)
        plt.show()
        self.mask = binary_closing(self.mask, square(25))
        self.mask = remove_small_objects(self.mask)
        fig, ax = plt.subplots()
        plt.imshow(self.mask)
        io.imsave('data/output/mask_with_borders_1.jpg', img_as_ubyte(self.mask))
        plt.close('all')
    def test8(self, y, x):
        slice = self.prepareSliceForDot(y, x)
        lengthsArray = self.convertSliceDotsToLenths(y, x, slice)
        # fig, ax = plt.subplots()
        # ax.bar(self.degreeArray, lengthsArray)
        # plt.show()
        if self.calculateVariationForLengthsArray(lengthsArray) > 40:
            return
        derivativeArray = self.prepareDerivativeArray(lengthsArray)
        new_lengths_array, new_derivative_array = self.process_dot_border(y, x, slice, lengthsArray, derivativeArray)
        self.drawBorderInMask(y, x, new_lengths_array, slice)

        # fig, ax = plt.subplots()
        # ax.bar(self.degreeArray, lengthsArray)
        # ax.plot(self.degreeArray, new_lengths_array, color='red')
        # plt.show()

        fig, ax = plt.subplots()
        plt.imshow(self.mask)
        plt.show()
    def test7(self, y, x):

        slice = (self.prepareSliceForDot(y, x))
        X = []
        Y = []
        for dot in slice:
            Y.append(dot[0])
            X.append(dot[1])
        fig, ax = plt.subplots()
        ax.scatter(X, Y, s=1, marker=",")
        ax.scatter([x], [y], s=1, marker=",")
        ax.imshow(self.mask)
        plt.show()

        lengthsArray = self.convertSliceDotsToLenths(y, x, slice)
        derivativeArray = self.prepareDerivativeArray(lengthsArray)
        # rolling_derivative = self.prepareRollingDerivativeArray(derivativeArray, 20)

        filter_size = 5
        exp_lengths_array = np.concatenate((lengthsArray[-filter_size:], lengthsArray, lengthsArray[:filter_size]))
        flattened_array_min = np.array([])
        flattened_array_mean = np.array([])
        for i in range(filter_size, len(lengthsArray)+filter_size):
            flattened_array_min = np.append(flattened_array_min, min(exp_lengths_array[(i-filter_size):(i+filter_size)]))
            flattened_array_mean = np.append(flattened_array_mean, exp_lengths_array[(i-filter_size):(i+filter_size)].mean())
        # flattened_array_min_derivative = self.prepareDerivativeArray(flattened_array_min)
        fig, ax = plt.subplots()
        ax.bar(self.degreeArray, lengthsArray)
        # ax.plot(self.degreeArray, flattened_array_min, color='red')
        ax.plot(self.degreeArray, flattened_array_mean, color='green')
        # ax.plot(self.degreeArray, (flattened_array_mean + flattened_array_min)/2, color='brown')
        ax.plot(self.degreeArray, derivativeArray, color='yellow')
        # ax.plot(self.degreeArray, flattened_array_min_derivative, color='orange')
        plt.show()
    def test3(self):
        y, x, = 350, 130
        slice = (self.prepareSliceForDot(y, x))
        lengthsArray = self.convertSliceDotsToLenths(y, x, slice)
        derivativeArray = self.prepareDerivativeArray(lengthsArray)
        # variation = self.calculateVariationForLengthsArray(lengthsArray)
        print(derivativeArray)
    def test2(self):
        y, x, = 140, 640

        slice = (self.prepareSliceForDot(y, x))
        lengthsArray = self.convertSliceDotsToLenths(y, x, slice)
        derivativeArray = self.prepareDerivativeArray(lengthsArray)
        print(self.calculateVariationForLengthsArray(lengthsArray))
        fig, ax = plt.subplots()
        ax.bar(self.degreeArray, lengthsArray)
        ax.plot(self.degreeArray, derivativeArray, color='red')
        plt.show()
    def test1(self):
        y, x, = 400, 270
        slice = (self.prepareSliceForDot(y, x))
        X = []
        Y = []
        for dot in slice:
            Y.append(dot[0])
            X.append(dot[1])
        fig, ax = plt.subplots()
        ax.scatter(X, Y, marker="s")
        ax.scatter([x], [y], marker="s")
        plt.imshow(self.mask)
        plt.show()
    def test0(self):
        X = []
        Y = []
        steps = 50

        for i in range(1, steps):
            x, y = self.move_to_border(300, i)
            X.append(x)
            Y.append(y)

        fig, ax = plt.subplots()
        ax.scatter(X, Y, marker="s")
        plt.show()
    def test5(self):
        for degree in self.degreeArray:
            y, x = self.move_to_border(degree, 5)

            fig, ax = plt.subplots()
            ax.set(xlim=(-6, 6), ylim=(-6, 6))
            a_circle = plt.Circle((0, 0), 5)
            ax.scatter([x], [y], color="red")
            ax.add_artist(a_circle)
            plt.title(degree)
            plt.show()
    def test6(self):
        mask = self.mask
        closer = square(10)
        rocks_img = io.imread('data/input/pred2.bmp')
        y_len = rocks_img.shape[0]
        x_len = rocks_img.shape[1]
        rocks_img = rocks_img[50:y_len-50]
        rocks_img = rocks_img[:, 50:x_len-50]
        new_mask = binary_closing(mask, closer)
        new_mask = remove_small_objects(new_mask)
        X = []
        Y = []
        for y, y_val in enumerate(new_mask):
            for x, x_val in enumerate(y_val):
                if x_val == True:
                    Y.append(y)
                    X.append(x)
        fig, ax = plt.subplots()
        io.imshow(rocks_img)
        ax.scatter(X, Y, s=1, marker=",")
        plt.show()
        new_mask = img_as_ubyte(new_mask)
        io.imsave('data/input/processed_mask_test.jpg', new_mask)
        exit()

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def linear(x, a, b):
    return a*x + b