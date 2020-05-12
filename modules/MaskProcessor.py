import numpy as np
import math
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
    degree_array = np.array([])

    # @var int Угол в градусах для поска обратного сильного изменения
    max_steps_to_closing = None

    def __init__(self, mask):
        """   Конструктор   """
        # @var mask numpy.array - Маска границ

        self.mask = mask
        self.prepare_degree_array()
        self.max_steps_to_closing = int(len(self.degree_array) / 6)
        self.mask_morphology_close(25)
        self.process_mask()
        self.closing_number = 2
        self.process_mask()

    def process_mask(self):
        """   Обработать маску   """
        shape = self.mask.shape
        for y in range(20, shape[0], 20):
            for x in range(20, shape[1], 20):
                if self.mask[y][x] == True:
                    continue
                slice = self.prepare_slice_for_dot(y, x)
                lengths_array = self.convert_slice_dots_to_lenths(y, x, slice)
                if self.calculate_variation_for_lengths_array(lengths_array) > self.VARIATION_LIMITS[self.closing_number]:
                    continue
                derivative_array = self.prepare_derivative_array(lengths_array)
                new_lengths_array, new_derivative_array = self.process_dot_border(y, x, lengths_array, derivative_array)
                self.draw_border_in_mask(y, x, new_lengths_array)
        self.mask_morphology_close(10)

    def prepare_degree_array(self):
        """   Подготовить градусы, по которым будет просчитываться угол   """
        degree = 0
        while degree < 360:
            self.degree_array = np.append(self.degree_array, degree)
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

    def prepare_slice_for_dot(self, dotY, dotX):
        """   Построить срез прострелов маски в определенной точке   """
        # @var dotY int - координата У
        # @var dotX int -  координата X
        # @return np.array массив точек прострелов для градусов

        limitsY = (0, self.mask.shape[0] - 1)
        limitsX = (0, self.mask.shape[1] - 1)
        slice = np.empty((0, 2), dtype=int)
        for degree in self.degree_array:
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

    def convert_slice_dots_to_lenths(self, dotY, dotX, slice):
        """   Конвертировать срез прострелов маски в определенной точке в массив длин  """
        # @var dotY int - координата У
        # @var dotX int -  координата X
        # @var np.array массив точек прострелов для градусов
        # @return np.array массив длин до границ по градусам

        lengths_array = np.array([])
        for dot in slice:
            lengthY = abs(abs(dot[0]) - dotY)
            lengthX = abs(abs(dot[1]) - dotX)
            length = (lengthY**2 + lengthX**2)**0.5
            lengths_array = np.append(lengths_array, length)
        return lengths_array

    def calculate_variation_for_lengths_array(self, lengths_array):
        """   Конвертировать срез прострелов маски в определенной точке в массив длин  """
        # @var np.array lengths_array массив длин до границ по градусам
        # @float вариация выборки
        new_lengths_array = deepcopy(lengths_array)
        new_lengths_array.sort()
        new_lengths_array = new_lengths_array[10:350]
        return new_lengths_array.std() / new_lengths_array.mean()

    def prepare_derivative_array(self, lengths_array):
        """   Получить массив производных для массива длин """
        # @var np.array lengths_array массив длин до границ по градусам
        # @return np.array массив производных для массива длин

        derivative_array = np.array([])
        length_of_array = lengths_array.shape[0]
        for index in range(length_of_array):
            if index + 1 < length_of_array:
                nextIndex = index + 1
            else:
                nextIndex = 0
            derivative_array = np.append(derivative_array, lengths_array[nextIndex] - lengths_array[index])
        return derivative_array

    def close_border(self, y, x, lengths_array, start, stop):
        """   Замкнуть границу """
        # @var np.array lengths_array массив длин до границ по градусам
        # @var int start Индекс началной координаты выброса
        # @var int stop Индекс конечной координаты воброса
        # @var int y Координата у точки
        # @var int x Координата х точки
        # @return (np.array массив длин лучей, np.array массив производных для массива длин лучей)

        end_lengths = 1 # длина массива перед и после интерполируемыми значениями
        degree_array_length = len(self.degree_array) # длина массива длин

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
            y_data = np.append(y_data, lengths_array[int(i)])

        popt, pcov = curve_fit(linear, x_data, y_data)
        a, b = popt

        for numb, i in enumerate(range(start + 1, start + range_length)):
            index = i
            if i >= degree_array_length:
                index -= degree_array_length
            length_value = linear(numb + end_lengths, a, b)
            lengths_array[int(index)] = round(length_value, 0)
        return lengths_array, self.prepare_derivative_array(lengths_array)

    def process_dot_border(self, y, x, lengths_array, derivative_array):        
        """   Обработать границы для точки """
        # @var np.array lengths_array массив длин до границ по градусам
        # @var np.array derivative_array массив производных длин до границ по градусам
        # @return (np.array массив длин лучей, np.array массив производных для массива длин лучей)

        limit = lengths_array.mean() * 0.3
        for index, degree in enumerate(self.degree_array):
            stop = False
            if derivative_array[index] > limit:
                stop = self.find_stop_by_outburst(index, derivative_array)
                if stop != False and self.closing_number == 1:
                    lengths_array, derivative_array = self.close_border(y, x, deepcopy(lengths_array), index, stop)
                if stop == False and self.closing_number == 2:
                    stop = self.find_stop_by_one_deviation(index, deepcopy(lengths_array))
                    if stop != False:
                        lengths_array, derivative_array = self.close_border(y, x, deepcopy(lengths_array), index, stop)
        return lengths_array, derivative_array

    def draw_border_in_mask(self, y, x, lengths_array):
        """   Отрисовать границы на маске для обработанной точки """
        # @var np.array lengths_array массив длин до границ по градусам
        # @var int y Координата у точки
        # @var int x Координата х точки

        for i, degree in enumerate(self.degree_array):
            border_y = y + math.sin(degree * math.pi / 180) * lengths_array[i]
            border_y = int(round(border_y))
            if border_y > self.mask.shape[0]:
                border_y = self.mask.shape[0]
            elif border_y < 0:
                border_y = 0
            border_x = x + math.cos(degree * math.pi / 180) * lengths_array[i]
            border_x = int(round(border_x))
            if border_x > self.mask.shape[1]:
                border_x = self.mask.shape[1]
            elif border_x < 0:
                border_x = 0
            self.mask[border_y-1:border_y+1, border_x-1:border_x+1] = True

    def find_stop_by_outburst(self, index, derivative_array):
        """   Найти окончание выброса по обратному выбросу """
        # @var np.array lengths_array массив длин до границ по градусам
        # @var int index Индекс началной координаты выброса
        # @return int|bool(false) - индес окончания выброса (если не найден - то False)

        stop_index = index + self.max_steps_to_closing # когда останавливаемся
        degrees_total = len(self.degree_array) # всего точек
        start_length = derivative_array[index] # стартовая длина
        while index < stop_index:
            if index < degrees_total:
                current_index = index
            else:
                current_index = index - degrees_total
            if start_length*self.CLOSURE_RATIO < -derivative_array[current_index] < start_length/self.CLOSURE_RATIO:
                return current_index
            index += 1
        return False

    def find_stop_by_one_deviation(self, index, legths_array):
        """   Найти окончание выброса по единственному отклонению """
        # @var np.array lengths_array массив длин до границ по градусам
        # @var int index Индекс началной координаты выброса
        # @return int|bool(false) - индес окончания выброса (если не найден - то False)

        degree_array_length = len(self.degree_array)  # длина массива углов
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

    def mask_morphology_close(self, thickness):
        self.mask = binary_closing(self.mask, square(thickness))
        self.mask = remove_small_objects(self.mask)


def linear(x, a, b):
    return a*x + b