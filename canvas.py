import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import colors

BRUSH_WEIGHT = 25
BRUSH_COLOR = colors.WHITE
BG_COLOR = colors.BLACK

class Canvas:

    def __init__(self, title: str ='draw image', filename: str ='image.png') -> None:
        self.title = title
        self.filename = f'images/{filename}'
        self.drawing = False
        self.image = np.full((500, 500, 3), BG_COLOR, dtype=np.uint8)
        self.ix = self.iy = 0
        self.brush_color = BRUSH_COLOR

    def set_xy(self, x: float, y: float) -> None:
        self.ix, self.iy = x, y

    def set_brush_color(self, color=None):
        if color:
            self.brush_color = color
        else:
            if self.brush_color == colors.BLACK:
                self.brush_color = colors.WHITE
            else:
                self.brush_color = colors.BLACK

    def paint_draw(self, event, x: float, y: float, flags, param) -> list:

        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.set_xy(x, y)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.line(self.image, (self.ix, self.iy), (x, y), color=self.brush_color, thickness=BRUSH_WEIGHT)
                self.set_xy(x, y)

        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = False
            cv.line(self.image, (self.ix, self.iy), (x, y), color=self.brush_color, thickness=BRUSH_WEIGHT)
            self.set_xy(x, y)

        return x, y

    def show_image(self, img):
        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.show()

    def draw_image(self):
        cv.namedWindow(self.title)
        cv.setMouseCallback(self.title, self.paint_draw)

        running = True
        while running:
            cv.imshow(self.title, self.image)
            
            k = cv.waitKey(1) & 0xFF
            if k == ord('t'): # toggle color
                self.set_brush_color()
            if k == ord('c'): # clear
                self.image = np.full((500, 500, 3), BG_COLOR, dtype=np.uint8)
            if k == ord('s'): # save
                cv.imwrite(self.filename, self.image)
            if k == 27 or k == ord('x'): # exit (ESCAPE)
                running = False

        cv.imwrite(self.filename, self.image)
        cv.destroyWindow(self.title)
        cv.destroyAllWindows()
        
        return self.image


def main() -> None:
    canvas = Canvas()
    img = canvas.draw_image()
    # canvas.show_image(img)

if __name__ == '__main__':
    main()
