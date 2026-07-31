import io
import tkinter as tk

from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def render_latex(self, text):
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, text)

        canvas = FigureCanvasAgg(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf)
        return ImageTk.PhotoImage(img)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        img = self.render_latex(self.text)
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("tahoma", "9", "normal"),
            padx=5,
            pady=2,
        )
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()
