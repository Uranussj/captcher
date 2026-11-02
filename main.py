
#!/usr/bin/env python3
"""
Captcha Reader - самостоятельное оконное приложение (без браузера).

Берёт картинку капчи из одного из трёх источников:
  1. файл с диска;
  2. буфер обмена (скопированная картинка);
  3. область экрана, которую вы выделяете мышью.

Дальше чистит изображение, распознаёт текст через Tesseract и показывает
результат. Кнопка "Копировать" кладёт текст в буфер обмена.

Рассчитано на простые текстовые капчи (свои сайты, тестовые стенды, учебные
задачи). Сложные капчи (reCAPTCHA, hCaptcha и т.п.) не поддерживаются.

Запуск:  python captcha_app.py
Сборка в .exe:  pyinstaller --onefile --windowed captcha_app.py
"""

import os
import re
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab, ImageTk

DEFAULT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PREVIEW_SIZE = (380, 110)  # максимальный размер превью


# --------------------------------------------------------------------------
# Поиск Tesseract (в Windows его часто нет в PATH)
# --------------------------------------------------------------------------
def locate_tesseract() -> bool:
    """Пытается найти tesseract; возвращает True, если нашли."""
    if shutil.which("tesseract"):
        return True
    candidates = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    return False


# --------------------------------------------------------------------------
# Обработка и распознавание
# --------------------------------------------------------------------------
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """Делает из картинки чёрный текст на белом фоне."""
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() < 127:  # тёмный фон -> инвертируем
        bw = cv2.bitwise_not(bw)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.copyMakeBorder(bw, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)


def recognize(processed: np.ndarray, alphabet: str) -> str:
    config = f"--oem 3 --psm 7 -c tessedit_char_whitelist={alphabet}"
    raw = pytesseract.image_to_string(processed, config=config)
    return re.sub(f"[^{re.escape(alphabet)}]", "", raw)


# --------------------------------------------------------------------------
# Выделение области экрана
# --------------------------------------------------------------------------
class ScreenSelector:
    """Полноэкранный оверлей: тянете мышью рамку - получаете вырезанную область."""

    def __init__(self, root: tk.Tk, on_done):
        self.root = root
        self.on_done = on_done
        self.start = None
        self.rect = None

        # Снимок всего экрана (окно приложения к этому моменту уже скрыто)
        self.shot = ImageGrab.grab()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        # На HiDPI-экранах размер снимка может отличаться от размера окна Tk
        self.kx = self.shot.width / sw
        self.ky = self.shot.height / sh

        self.top = tk.Toplevel(root)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-topmost", True)
        self.top.configure(cursor="crosshair")
        self.canvas = tk.Canvas(self.top, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.photo = ImageTk.PhotoImage(self.shot.resize((sw, sh)))
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.create_text(
            sw // 2, 30, fill="red", font=("Arial", 16, "bold"),
            text="Выделите капчу мышью (Esc - отмена)",
        )

        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.top.bind("<Escape>", lambda e: self.finish(None))

    def press(self, e):
        self.start = (e.x, e.y)
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)

    def drag(self, e):
        if self.rect:
            self.canvas.coords(self.rect, *self.start, e.x, e.y)

    def release(self, e):
        if not self.start:
            return
        x1, x2 = sorted((self.start[0], e.x))
        y1, y2 = sorted((self.start[1], e.y))
        if x2 - x1 < 5 or y2 - y1 < 5:  # случайный клик
            self.finish(None)
            return
        box = (int(x1 * self.kx), int(y1 * self.ky), int(x2 * self.kx), int(y2 * self.ky))
        self.finish(self.shot.crop(box))

    def finish(self, image):
        self.top.destroy()
        self.on_done(image)


# --------------------------------------------------------------------------
# Главное окно
# --------------------------------------------------------------------------
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Captcha Reader")
        root.resizable(False, False)

        self.image: Image.Image | None = None
        self._photos = []  # храним ссылки, иначе Tk "съест" картинки

        main = ttk.Frame(root, padding=12)
        main.grid()

        # Кнопки источников
        src = ttk.Frame(main)
        src.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(src, text="Открыть файл…", command=self.load_file).pack(side="left", padx=2)
        ttk.Button(src, text="Из буфера обмена", command=self.load_clipboard).pack(side="left", padx=2)
        ttk.Button(src, text="Область экрана", command=self.grab_screen).pack(side="left", padx=2)

        # Превью оригинала и обработанного изображения
        ttk.Label(main, text="Оригинал").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.orig_label = ttk.Label(main, relief="sunken")
        self.orig_label.grid(row=2, column=0, columnspan=2, sticky="ew")

        ttk.Label(main, text="После обработки").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.proc_label = ttk.Label(main, relief="sunken")
        self.proc_label.grid(row=4, column=0, columnspan=2, sticky="ew")

        # Настройки
        ttk.Label(main, text="Допустимые символы:").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.alphabet = tk.StringVar(value=DEFAULT_ALPHABET)
        ttk.Entry(main, textvariable=self.alphabet, width=50).grid(row=6, column=0, columnspan=2, sticky="ew")

        # Быстрые пресеты алфавита
        presets = ttk.Frame(main)
        presets.grid(row=7, column=0, columnspan=2, sticky="w", pady=2)
        for name, value in (
            ("Цифры", "0123456789"),
            ("Заглавные+цифры", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            ("Всё", DEFAULT_ALPHABET),
        ):
            ttk.Button(presets, text=name, command=lambda v=value: self.alphabet.set(v)).pack(side="left", padx=2)

        ttk.Label(main, text="Ожидаемая длина (0 = любая):").grid(row=8, column=0, sticky="w", pady=(8, 0))
        self.length = tk.IntVar(value=0)
        ttk.Spinbox(main, from_=0, to=20, textvariable=self.length, width=5).grid(row=8, column=1, sticky="w", pady=(8, 0))

        ttk.Button(main, text="Распознать", command=self.run).grid(row=9, column=0, columnspan=2, pady=10, sticky="ew")

        # Результат
        self.result = tk.StringVar()
        ttk.Entry(main, textvariable=self.result, font=("Consolas", 20), justify="center").grid(
            row=10, column=0, columnspan=2, sticky="ew"
        )
        ttk.Button(main, text="Копировать", command=self.copy).grid(row=11, column=0, columnspan=2, pady=(6, 0), sticky="ew")

        self.status = tk.StringVar(value="Выберите источник картинки")
        ttk.Label(main, textvariable=self.status, foreground="gray").grid(row=12, column=0, columnspan=2, pady=(8, 0), sticky="w")

    # ---- показ картинок -------------------------------------------------
    @staticmethod
    def fit(img: Image.Image) -> Image.Image:
        """Уменьшает картинку под область превью (увеличивать не будем)."""
        img = img.copy()
        img.thumbnail(PREVIEW_SIZE)
        return img

    def show(self, label: ttk.Label, img: Image.Image):
        photo = ImageTk.PhotoImage(self.fit(img))
        self._photos.append(photo)
        self._photos = self._photos[-4:]
        label.configure(image=photo)

    def set_image(self, img: Image.Image | None):
        if img is None:
            self.status.set("Отменено")
            return
        self.image = img
        self.show(self.orig_label, img)
        self.proc_label.configure(image="")
        self.result.set("")
        self.status.set("Картинка загружена. Нажмите «Распознать».")
        self.run()  # сразу пробуем распознать

    # ---- источники ------------------------------------------------------
    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        try:
            self.set_image(Image.open(path))
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{exc}")

    def load_clipboard(self):
        data = ImageGrab.grabclipboard()
        if isinstance(data, Image.Image):
            self.set_image(data)
        elif isinstance(data, list) and data:  # скопирован файл
            try:
                self.set_image(Image.open(data[0]))
            except Exception:
                messagebox.showinfo("Буфер обмена", "В буфере нет картинки.")
        else:
            messagebox.showinfo("Буфер обмена", "В буфере нет картинки.")

    def grab_screen(self):
        self.root.withdraw()  # прячем окно, чтобы оно не попало на снимок
        self.root.after(300, lambda: ScreenSelector(self.root, self.after_select))

    def after_select(self, img):
        self.root.deiconify()
        self.set_image(img)

    # ---- распознавание --------------------------------------------------
    def run(self):
        if self.image is None:
            messagebox.showinfo("Нет картинки", "Сначала загрузите картинку капчи.")
            return
        alphabet = self.alphabet.get().strip() or DEFAULT_ALPHABET
        try:
            processed = preprocess(self.image)
            self.show(self.proc_label, Image.fromarray(processed))
            text = recognize(processed, alphabet)
        except pytesseract.TesseractNotFoundError:
            messagebox.showerror(
                "Tesseract не найден",
                "Установите Tesseract OCR и добавьте его в PATH\n"
                "или задайте переменную окружения TESSERACT_CMD.",
            )
            return
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))
            return

        self.result.set(text)
        want = self.length.get()
        if not text:
            self.status.set("Текст не распознан. Попробуйте другой алфавит или картинку.")
        elif want and len(text) != want:
            self.status.set(f"Распознано {len(text)} символов, ожидалось {want} - возможна ошибка.")
        else:
            self.status.set("Готово.")

    def copy(self):
        text = self.result.get()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status.set("Скопировано в буфер обмена.")


def main():
    root = tk.Tk()
    if not locate_tesseract():
        messagebox.showwarning(
            "Tesseract не найден",
            "Tesseract OCR не обнаружен. Установите его и перезапустите программу\n"
            "(или задайте путь в переменной окружения TESSERACT_CMD).",
        )
    App(root)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())