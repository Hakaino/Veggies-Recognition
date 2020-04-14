import cv2
import numpy


# This function returns a binary version emphasizing the prominent items of a image passed as argument
def segmentation(image):
    # First the image is turn gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Then a threshold is applied, attempting to highlight the vegetables
    ret, thresh = cv2.threshold(gray, 120, 250, cv2.THRESH_BINARY)
    # then the morphology methods to open then close holes will reduce noise,
    # as well as solidifying the shape of the vegetables
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    open_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel)
    return close_img


# This function identifies different item types in a image, by analysing their size shape and color.
# Then returns a dictionary and a contour list
def classification(image, original):
    dictionary = {}
    # search for contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in range(len(contours)):
        # Only the contours with a certain area are analyzed
        if 10 ** 4 < cv2.contourArea(contours[contour]) < 10 ** 7:
            # the ratio between with and height of the minimum bounding box is defined,
            # as it can be used as a classification method
            rect = cv2.minAreaRect(contours[contour])
            ratio = min(rect[1]) / max(rect[1])
            # carrots have the smallest ration of the considered vegetables
            if ratio < .4:
                # The index in the list "contours" and type of the item
                # represented by that contour are set in the dictionary
                dictionary.setdefault(contour, "carrot")
            # after, is the ratio of the babbage
            elif ratio < .6:
                dictionary.setdefault(contour, "cabbage")
            # both potatoes and bell peppers have the biggest ratios of the group, but the values are very close to
            # each other, so an other method of classification is required
            else:
                # A mask made from a specific contour's shape to apply to the original image, isolating the vegetable,
                # in order to calculate it's colour mean value
                mask = numpy.zeros(image.shape, numpy.uint8)
                cv2.drawContours(mask, contours, contour, (255, 255, 255), -1)
                mean_color = cv2.mean(original, mask=mask)
                # the largest difference in colour between the potatoes and the bell peppers are the red and green,
                # therefore a colour region is selected for the potatoes
                if 90 < mean_color[1] and 100 < mean_color[2]:
                    dictionary.setdefault(contour, "potato")
                # and the remaining vegetables are defined as bell peppers
                else:
                    dictionary.setdefault(contour, "bell pepper")
    return dictionary, contours


# This function prints the already classified contours on a given image,
# with color code differentiation per item type.
def printContours(image, dictionary, contours):
    # "contours" is a list of all the contours found
    # "dictionary" relates the index in "contours" with the type of item represented there.
    # Only the identified items are an "entry" in the "dictionary"
    for entry in dictionary:
        if dictionary.get(entry) == "carrot":
            colour = (0, 140, 255)  # "colour" is a sequence of 8bit values [0-255] set in blue; green; red (BGR) order
        elif dictionary.get(entry) == "cabbage":
            colour = (0, 255, 0)
        elif dictionary.get(entry) == "potato":
            colour = (29, 70, 110)
        elif dictionary.get(entry) == "bell pepper":
            colour = (19, 69, 239)
        else:
            # if a entry in the dictionary is not correctly identified,
            # the function returns false without ever printing the contours in the image
            raise Exception("printContours: A entry in the dictionary is not correctly identified")
        # Draws the contour with index "entry" in the list "contours"
        # the "colour" of the trace was previously defined and 50 is the thickness of the trace in pixels
        cv2.drawContours(image, contours, entry, colour, 50)


# This function displays an image
def show(image, width=500, time=0):
    # The "width" is the value in pixels of the width a proportional resized version of the image to be displayed.
    # This is done so that the image can fit the screen, before it is displayed
    height = int(width * image.shape[0] / image.shape[1])
    resize = cv2.resize(image, (width, height))
    cv2.imshow('final image', resize)
    # "time" is the waiting period, defined in milliseconds.
    # If time is set to 0 it will wait until a key is pressed,
    # before closing the window with the displayed image
    cv2.waitKey(time)
    cv2.destroyAllWindows()


# Here is where the program starts
if __name__ == '__main__':
    # From the point of view of the image processing pipeline, thought in Robotic programing
    # The image was acquired through the camera of a smartphone
    img = cv2.imread('veggies.jpg', 1)
    # segmentation(), first performs pre-processing and then segments the image
    segmented = segmentation(img)
    # classification(), first represents the object then classifies it
    d, c = classification(segmented, img)
    printContours(img, d, c)
    show(img)
