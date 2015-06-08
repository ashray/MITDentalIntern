import cv2

def image_downscale(input_image, max_dimension):
    height, width, depth = input_image.shape
    height, width = width, height
    if height > width:
        dim = (max_dimension, int(((width * max_dimension) / height)))
        input_image = cv2.resize(input_image, dim)  # , interpolation = cv2.INTER_AREA)
    else:
        dim = (int(((height * max_dimension) / width)), max_dimension)
    input_image = cv2.resize(input_image, dim)  # , interpolation = cv2.INTER_AREA)
    #
    # cv2.imshow('input_image', input_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite('downscaled_image.png', input_image)
    return input_image