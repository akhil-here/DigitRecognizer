import cv2
import numpy as np
from tkinter import *
from PIL import ImageGrab, ImageTk
from keras.models import load_model
model = load_model('mnist.h5')  # load the model

source = Tk()  # Create the main window
source.resizable(0, 0)
source.geometry("300x235")
source.title("Digit Recognition")
initx, inity = None, None
image_number = 0


def clear_source():
    global draw_area
    draw_area.delete("all")


def activate_event(event):
    global initx, inity
    draw_area.bind('<B1-Motion>', draw_lines)
    initx, inity = event.x, event.y


def draw_lines(event):
    global initx, inity
    x, y = event.x, event.y
    draw_area.create_line((initx, inity, x, y), width=7,
                          fill='black', capstyle=ROUND, smooth=True, splinesteps=12)
    initx, inity = x, y


def Recognize_Digit():
    global image_number
    filename = f'image_{image_number}.png'
    widget = draw_area
    x = source.winfo_rootx() + widget.winfo_x()
    y = source.winfo_rooty() + widget.winfo_y()
    x1 = x+widget.winfo_width()
    y1 = y+widget.winfo_height()

    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    digit = cv2.imread(filename, cv2.IMREAD_COLOR)
    make_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(
        make_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
        0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(digit, (x, y), (x+w, y+h), (255, 0, 0), 1)
        top = int(0.05*th.shape[0])
        bottom = top
        left = int(0.05*th.shape[1])
        right = left
        roi = th[y-top:y+h+bottom, x-left:x+w+right]
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        img = img.reshape(1, 28, 28, 1)
        img = img/255.0
        prediction = model.predict([img])[0]
        final = np.argmax(prediction)
    label = Label(source, text=final, font=("Arial", 25))
    label.config(font=("Courier", 100))
    label.grid(row=0, column=3, pady=1, padx=1)


draw_area = Canvas(source, width=150, height=200, bg='white')
draw_area.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

draw_area.bind('<Button-1>', activate_event)
sto = Style()
sto.configure('W.RButton', font= ('Arial', 10, 'underline'),
foreground='Green')
sto.configure('W.CButton', font= ('Arial', 10, 'underline'),
foreground='Red')
btn_save = Button(text="Recognize the Digit",
                  style='W.RButton', command=Recognize_Digit)
btn_save.grid(row=3, column=0, pady=1, padx=1, columnspan=2)
button_clear = Button(text="Clear Area", style='W.CButton', command=clear_source)
button_clear.grid(row=3, column=2, pady=1, padx=1, columnspan=3)

source.mainloop()
