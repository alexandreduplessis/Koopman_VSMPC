import numpy as np

def draw_centered_square(x, y, size, length, width):
    """ Draw a square in an array of size (size, size) centered on (x, y) """
    array = np.zeros((length, width))
    array[x-size:x+size, y-size:y+size] = 1
    return array

def find_max_size_square_center_coordinates(array):
    """ Find the center of the largest square in an array """
    max_size = 0
    max_x = 0
    max_y = 0
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            if array[x, y] == 1:
                size = 1
                while True:
                    if x+size >= array.shape[0] or y+size >= array.shape[1]:
                        break
                    if np.all(array[x:x+size, y:y+size] == 1):
                        size += 1
                    else:
                        break
                if size > max_size:
                    max_size = size
                    max_x = x
                    max_y = y
    return max_x, max_y, max_size

def loss_to_reward(loss):
    """ Convert a loss to a reward """
    return 100./(loss+1.)