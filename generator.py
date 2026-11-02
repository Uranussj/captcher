#!/usr/bin/env python3
"""
Генератор учебных капч для обучения нейросети (train.py).

Программа сама рисует капчи (случайный текст, шрифт, фон, поворот букв,
волнистое искажение, линии, шум), показывает их на "табло" и сохраняет в папку
с именами вида  ОТВЕТ_xxxxxxxx.png  - именно такой формат читает train.py.

Запуск с окном:
    python generator_app.py

Без окна (быстро, из консоли):
    python generator_app.py --cli --count 3000 --out dataset_synth

Как подогнать стиль под ваши капчи:
    положите .ttf/.otf шрифты, похожие на ваши, в папку  fonts/  рядом с
    программой - тогда будут использоваться только они (иначе берутся системные).

Только Pillow + numpy + OpenCV (уже стоят для остальных программ), без новых зависимостей.
"""

import argparse
import random
import string
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH, HEIGHT = 220, 80
DEFAULT_ALPHABET = string.ascii_letters + string.digits
USER_FONTS_DIR = Path(__file__).resolve().parent / "fonts"

SYSTEM_FONT_DIRS = [
    r"C:\Windows\Fonts",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    str(Path.home() / ".fonts"),
    str(Path.home() / "Library" / "Fonts"),
]
# Служебные шрифты (значки, эмодзи) - буквы они не рисуют
BAD_FONT_WORDS = ("symbol", "wingding", "webding", "dingbat", "marlett", "emoji",
                  "holomdl", "segmdl", "bookshelf", "mtextra", "icon")

# Параметры "сложности": 0 - чисто ... 3 - тяжёлая капча
LEVELS = {
    0: dict(rot=6,  wave=0.0, lines=0, dots=0,   noise=3,  spread=1.00),
    1: dict(rot=12, wave=1.5, lines=1, dots=0,   noise=6,  spread=1.00),
    2: dict(rot=20, wave=2.5, lines=3, dots=150, noise=10, spread=0.92),
    3: dict(rot=28, wave=3.5, lines=5, dots=400, noise=14, spread=0.82),
}
LEVEL_NAMES = ["0 - чисто", "1 - лёгкая", "2 - средняя", "3 - сложная"]


# --------------------------------------------------------------------------
# Шрифты
# --------------------------------------------------------------------------
_font_cache = {}


def font_is_usable(path: str, alphabet: str) -> bool:
    """Шрифт подходит, если рисует все нужные символы (и отличает a от A)."""
    try:
        font = ImageFont.truetype(path, 40)
        for ch in alphabet:
            if font.getmask(ch).getbbox() is None:  # символа в шрифте нет
                return False
        if "a" in alphabet and "A" in alphabet and font.getbbox("a") == font.getbbox("A"):
            return False  # шрифт только из заглавных - строчные буквы были бы "враньём"
        return True
    except Exception:
        return False


def find_fonts(alphabet: str, fonts_dir=None) -> list:
    """Возвращает список путей к подходящим шрифтам (из fonts/ или системные)."""
    key = (alphabet, str(fonts_dir))
    if key in _font_cache:
        return _font_cache[key]

    dirs = [Path(fonts_dir)] if fonts_dir else []
    if not dirs and USER_FONTS_DIR.is_dir():
        dirs = [USER_FONTS_DIR]
    if not dirs or not any(p.suffix.lower() in (".ttf", ".otf") for d in dirs for p in d.rglob("*")):
        dirs = [Path(d) for d in SYSTEM_FONT_DIRS if Path(d).is_dir()]

    result = []
    for d in dirs:
        for p in d.rglob("*"):
            if p.suffix.lower() not in (".ttf", ".otf"):
                continue
            if any(w in p.stem.lower() for w in BAD_FONT_WORDS):
                continue
            if font_is_usable(str(p), alphabet):
                result.append(str(p))
    _font_cache[key] = result
    return result


# --------------------------------------------------------------------------
# Рисование капчи
# --------------------------------------------------------------------------
def make_background(size, rng: random.Random, noise_sigma: float) -> Image.Image:
    """Однотонный фон со случайным цветом, мягкими пятнами и зернистой текстурой."""
    w, h = size
    nrng = np.random.default_rng(rng.getrandbits(32))
    base = np.array([rng.randint(40, 255) for _ in range(3)], dtype=np.float32)

    blotches = nrng.normal(0, 16, (3, 5, 1)).astype(np.float32)  # крупные пятна яркости
    blotches = cv2.resize(blotches, (w, h), interpolation=cv2.INTER_CUBIC)[:, :, None]
    grain = nrng.normal(0, noise_sigma, (h, w, 1)).astype(np.float32)  # мелкое зерно
    arr = base[None, None, :] + blotches + grain
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def pick_text_color(bg: Image.Image, rng: random.Random):
    """Цвет текста, заметно отличающийся по яркости от фона."""
    mean = np.array(bg).reshape(-1, 3).mean(axis=0)
    lum = 0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2]
    if lum > 130:
        return tuple(rng.randint(0, 70) for _ in range(3))
    return tuple(rng.randint(185, 255) for _ in range(3))


def wave_distort(layer: Image.Image, amp: float, rng: random.Random) -> Image.Image:
    """Волнистое искажение слоя с текстом."""
    if amp <= 0:
        return layer
    arr = np.array(layer)
    h, w = arr.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wl = rng.uniform(35, 70)
    ph = rng.uniform(0, 6.28)
    map_y = ys + amp * np.sin(xs * 2 * np.pi / wl + ph)
    map_x = xs + amp * 0.7 * np.sin(ys * 2 * np.pi / (wl * 0.8) + ph * 1.7)
    out = cv2.remap(arr, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return Image.fromarray(out, "RGBA")


def make_captcha(text: str, font_path: str, size, level: int, rng: random.Random) -> Image.Image:
    w, h = size
    p = LEVELS[level]

    bg = make_background(size, rng, p["noise"])
    color = pick_text_color(bg, rng)

    n = len(text)
    margin = w * 0.05
    slot = (w - 2 * margin) / n
    font_px = int(min(h * rng.uniform(0.66, 0.80), slot * 1.30))
    font = ImageFont.truetype(font_path, max(font_px, 12))

    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    tile_w, tile_h = int(slot * 2.0) + 10, h
    for i, ch in enumerate(text):
        tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
        shade = tuple(int(np.clip(c + rng.randint(-12, 12), 0, 255)) for c in color)
        ImageDraw.Draw(tile).text((tile_w / 2, tile_h / 2), ch, font=font, fill=shade + (255,), anchor="mm")
        tile = tile.rotate(rng.uniform(-p["rot"], p["rot"]), resample=Image.BICUBIC)

        cx = margin + slot * (i + 0.5)
        cx = w / 2 + (cx - w / 2) * p["spread"]  # на высоких уровнях буквы сближаются и слипаются
        x = int(cx - tile_w / 2 + rng.uniform(-slot * 0.10, slot * 0.10))
        y = int(rng.uniform(-h * 0.06, h * 0.06))
        layer.paste(tile, (x, y), tile)

    layer = wave_distort(layer, p["wave"], rng)
    img = bg.convert("RGBA")
    img.alpha_composite(layer)
    img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    for _ in range(p["lines"]):  # линии поверх текста
        lc = color if rng.random() < 0.6 else tuple(rng.randint(0, 255) for _ in range(3))
        draw.line(
            [(rng.randint(0, w // 3), rng.randint(0, h)), (rng.randint(2 * w // 3, w), rng.randint(0, h))],
            fill=lc, width=rng.choice((1, 1, 2)),
        )
    for _ in range(p["dots"]):  # шум-точки
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        draw.point((x, y), fill=tuple(rng.randint(0, 255) for _ in range(3)))

    if level >= 1:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    return img


def generate_one(rng, fonts, alphabet, min_len, max_len, level, size=(WIDTH, HEIGHT)):
    """Возвращает (ответ, картинка)."""
    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))
    return text, make_captcha(text, rng.choice(fonts), size, level, rng)


def save_sample(out_dir: Path, text: str, img: Image.Image) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{text}_{uuid.uuid4().hex[:8]}.png"
    img.save(path)
    return path


# --------------------------------------------------------------------------
# Режим без окна
# --------------------------------------------------------------------------
def run_cli(args) -> int:
    alphabet = "".join(dict.fromkeys(args.alphabet))  # без повторов, порядок сохраняется
    fonts = find_fonts(alphabet, args.fonts)
    if not fonts:
        print("Не найдено подходящих шрифтов. Положите .ttf/.otf в папку fonts/ или укажите --fonts.")
        return 1
    print(f"Шрифтов: {len(fonts)}. Генерирую {args.count} капч в {args.out} ...")

    rng = random.Random(args.seed)
    out = Path(args.out)
    for i in range(1, args.count + 1):
        text, img = generate_one(rng, fonts, alphabet, args.min_len, args.max_len, args.level)
        save_sample(out, text, img)
        if i % 250 == 0 or i == args.count:
            print(f"  {i}/{args.count}")
    print("Готово. Дальше:")
    print(f'  python train.py train --data {args.out} --alphabet "{alphabet}" --epochs 30')
    return 0


# --------------------------------------------------------------------------
# Окно с табло
# --------------------------------------------------------------------------
def run_gui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import ImageTk

    class GeneratorApp:
        def __init__(self, root):
            self.root = root
            root.title("Генератор капч")
            root.resizable(False, False)
            self.rng = random.Random()
            self.fonts_dir = None
            self.fonts = []
            self.current = None  # (текст, картинка)
            self.photo = None
            self.busy = False

            f = ttk.Frame(root, padding=12)
            f.grid()

            # --- табло ---
            self.board = tk.Label(f, bg="#222222", width=48, height=9, relief="sunken")
            self.board.grid(row=0, column=0, columnspan=3, sticky="ew")
            self.answer = tk.StringVar(value="—")
            ttk.Label(f, textvariable=self.answer, font=("Consolas", 26, "bold"), anchor="center").grid(
                row=1, column=0, columnspan=3, sticky="ew", pady=(6, 8)
            )

            # --- настройки ---
            ttk.Label(f, text="Допустимые символы:").grid(row=2, column=0, columnspan=3, sticky="w")
            self.alphabet = tk.StringVar(value=DEFAULT_ALPHABET)
            ttk.Entry(f, textvariable=self.alphabet, width=60).grid(row=3, column=0, columnspan=3, sticky="ew")
            pre = ttk.Frame(f)
            pre.grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
            for name, val in (("Цифры", string.digits), ("Буквы", string.ascii_letters),
                              ("Заглавные+цифры", string.ascii_uppercase + string.digits), ("Всё", DEFAULT_ALPHABET)):
                ttk.Button(pre, text=name, command=lambda v=val: self.set_alphabet(v)).pack(side="left", padx=2)

            row = ttk.Frame(f)
            row.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))
            ttk.Label(row, text="Длина от").pack(side="left")
            self.min_len = tk.IntVar(value=5)
            ttk.Spinbox(row, from_=1, to=12, textvariable=self.min_len, width=4).pack(side="left", padx=4)
            ttk.Label(row, text="до").pack(side="left")
            self.max_len = tk.IntVar(value=6)
            ttk.Spinbox(row, from_=1, to=12, textvariable=self.max_len, width=4).pack(side="left", padx=4)
            ttk.Label(row, text="  Сложность").pack(side="left")
            self.level = ttk.Combobox(row, values=LEVEL_NAMES, state="readonly", width=12)
            self.level.current(2)
            self.level.pack(side="left", padx=4)

            frow = ttk.Frame(f)
            frow.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
            self.fonts_label = tk.StringVar()
            ttk.Label(frow, textvariable=self.fonts_label, foreground="gray").pack(side="left")
            ttk.Button(frow, text="Папка со шрифтами…", command=self.choose_fonts).pack(side="right")

            # --- действия ---
            self.btn_new = ttk.Button(f, text="Новая капча", command=self.new_captcha)
            self.btn_new.grid(row=7, column=0, sticky="ew", pady=(10, 0), padx=(0, 3))
            self.btn_save = ttk.Button(f, text="Сохранить эту", command=self.save_current)
            self.btn_save.grid(row=7, column=1, columnspan=2, sticky="ew", pady=(10, 0), padx=(3, 0))

            ttk.Separator(f).grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)
            ttk.Label(f, text="Папка для сохранения:").grid(row=9, column=0, sticky="w")
            self.out_dir = tk.StringVar(value="dataset_synth")
            ttk.Entry(f, textvariable=self.out_dir, width=30).grid(row=10, column=0, columnspan=2, sticky="ew")
            ttk.Button(f, text="…", width=3, command=self.choose_out).grid(row=10, column=2, sticky="w", padx=3)

            brow = ttk.Frame(f)
            brow.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(8, 0))
            ttk.Label(brow, text="Сколько:").pack(side="left")
            self.count = tk.IntVar(value=2000)
            ttk.Spinbox(brow, from_=1, to=200000, increment=500, textvariable=self.count, width=8).pack(side="left", padx=4)
            self.btn_batch = ttk.Button(brow, text="Сгенерировать пачку", command=self.start_batch)
            self.btn_batch.pack(side="left", padx=6)

            self.progress = ttk.Progressbar(f, maximum=100)
            self.progress.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(8, 0))
            self.status = tk.StringVar(value="Нажмите «Новая капча» или сразу «Сгенерировать пачку».")
            ttk.Label(f, textvariable=self.status, foreground="gray", wraplength=440).grid(
                row=13, column=0, columnspan=3, sticky="w", pady=(6, 0)
            )

            self.reload_fonts()
            self.new_captcha()

        # ---- вспомогательное ----
        def set_alphabet(self, value):
            self.alphabet.set(value)
            self.reload_fonts()

        def get_params(self):
            alphabet = "".join(dict.fromkeys(self.alphabet.get().strip()))
            lo, hi = self.min_len.get(), self.max_len.get()
            if not alphabet:
                raise ValueError("Укажите допустимые символы.")
            if lo < 1 or hi < lo:
                raise ValueError("Длина: «от» должно быть не меньше 1 и не больше «до».")
            return alphabet, lo, hi, self.level.current()

        def reload_fonts(self):
            alphabet = "".join(dict.fromkeys(self.alphabet.get().strip()))
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            self.fonts = find_fonts(alphabet, self.fonts_dir) if alphabet else []
            self.root.config(cursor="")
            src = "fonts/" if (self.fonts_dir or USER_FONTS_DIR.is_dir()) else "системные"
            self.fonts_label.set(f"Подходящих шрифтов: {len(self.fonts)} ({src})")

        def choose_fonts(self):
            d = filedialog.askdirectory(title="Папка со шрифтами (.ttf/.otf)")
            if d:
                self.fonts_dir = d
                self.reload_fonts()

        def choose_out(self):
            d = filedialog.askdirectory(title="Куда сохранять капчи")
            if d:
                self.out_dir.set(d)

        def show(self, text, img):
            self.current = (text, img)
            big = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)  # крупнее для табло
            self.photo = ImageTk.PhotoImage(big)
            self.board.configure(image=self.photo, width=big.width, height=big.height)
            self.answer.set(text)

        def set_busy(self, busy):
            self.busy = busy
            state = "disabled" if busy else "normal"
            for b in (self.btn_new, self.btn_save, self.btn_batch):
                b.configure(state=state)

        # ---- действия ----
        def new_captcha(self):
            try:
                alphabet, lo, hi, level = self.get_params()
            except ValueError as exc:
                messagebox.showwarning("Настройки", str(exc))
                return
            if not self.fonts:
                self.reload_fonts()
            if not self.fonts:
                messagebox.showwarning("Шрифты", "Нет подходящих шрифтов для этого набора символов.\n"
                                       "Положите .ttf/.otf в папку fonts/ или выберите папку кнопкой.")
                return
            self.show(*generate_one(self.rng, self.fonts, alphabet, lo, hi, level))

        def save_current(self):
            if not self.current:
                return
            path = save_sample(Path(self.out_dir.get()), *self.current)
            self.status.set(f"Сохранено: {path}")

        def start_batch(self):
            try:
                alphabet, lo, hi, level = self.get_params()
                total = self.count.get()
            except (ValueError, tk.TclError) as exc:
                messagebox.showwarning("Настройки", str(exc))
                return
            self.reload_fonts()
            if not self.fonts:
                messagebox.showwarning("Шрифты", "Нет подходящих шрифтов для этого набора символов.")
                return
            self.batch = dict(done=0, total=total, alphabet=alphabet, lo=lo, hi=hi, level=level,
                              out=Path(self.out_dir.get()))
            self.progress.configure(maximum=total, value=0)
            self.set_busy(True)
            self.batch_step()

        def batch_step(self):
            b = self.batch
            for _ in range(min(20, b["total"] - b["done"])):  # маленькая порция, чтобы окно не зависало
                text, img = generate_one(self.rng, self.fonts, b["alphabet"], b["lo"], b["hi"], b["level"])
                save_sample(b["out"], text, img)
                b["done"] += 1
            self.show(text, img)  # на табло - последняя созданная капча
            self.progress.configure(value=b["done"])
            self.status.set(f"Создано {b['done']} из {b['total']} -> {b['out']}")
            if b["done"] < b["total"]:
                self.root.after(1, self.batch_step)
            else:
                self.set_busy(False)
                self.status.set(
                    f"Готово: {b['total']} капч в «{b['out']}». Обучение:\n"
                    f'python train.py train --data {b["out"]} --alphabet "{b["alphabet"]}" --epochs 30'
                )

    root = tk.Tk()
    GeneratorApp(root)
    root.mainloop()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Генератор учебных капч")
    p.add_argument("--cli", action="store_true", help="работать без окна")
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--out", default="dataset_synth")
    p.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--max-len", type=int, default=6)
    p.add_argument("--level", type=int, default=2, choices=sorted(LEVELS))
    p.add_argument("--fonts", help="папка со шрифтами (по умолчанию fonts/ или системные)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    return run_cli(args) if args.cli else run_gui()


if __name__ == "__main__":
    sys.exit(main())