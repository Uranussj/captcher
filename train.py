#!/usr/bin/env python3
"""
Обучаемое распознавание текстовых капч (нейросеть CRNN + CTC, PyTorch).

Режимы:
  1) Обучение на синтетических капчах (учебный режим, данные генерирует
     библиотека `captcha`, ничего размечать не нужно):
         python train.py train --synthetic --epochs 15

  2) Обучение на ваших картинках из папки dataset/.
     Имя файла = ответ + "_" + что угодно:  K7P2X_a1b2c3.png  ->  "K7P2X"
         python train.py train --data dataset

  3) Дообучение уже готовой модели на новых примерах (программа "учится"):
         python train.py train --data dataset --resume model.pt

  4) Проверка на одной картинке:
         python train.py predict --model model.pt captcha.png

Используйте только для своих капч, тестовых стендов и учебных задач.
"""

import argparse
import random
import string
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

H, W = 32, 128  # размер входа сети (высота, ширина)
DEFAULT_ALPHABET = string.ascii_uppercase + string.digits


# --------------------------------------------------------------------------
# Подготовка картинки
# --------------------------------------------------------------------------
def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL-картинка -> тензор 1xHxW со значениями 0..1."""
    img = img.convert("L").resize((W, H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------
class FolderDataset(Dataset):
    """Картинки из папки; ответ - часть имени файла до первого '_'."""

    def __init__(self, folder: str, alphabet: str):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
        self.items = []
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() not in exts:
                continue
            label = p.stem.split("_")[0]
            if label and all(c in alphabet for c in label):
                self.items.append((p, label))
        if not self.items:
            raise SystemExit(f"В папке {folder} нет подходящих картинок (формат имени: ОТВЕТ_что-угодно.png)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return to_tensor(Image.open(path)), label


class SyntheticDataset(Dataset):
    """Генерирует капчи на лету библиотекой `captcha` (для обучения и экспериментов)."""

    def __init__(self, size: int, alphabet: str, min_len=4, max_len=6, seed=None):
        try:
            from captcha.image import ImageCaptcha
        except ImportError:
            raise SystemExit("Для --synthetic установите библиотеку: pip install captcha")
        self.gen = ImageCaptcha(width=160, height=60)
        self.size, self.alphabet = size, alphabet
        self.min_len, self.max_len = min_len, max_len
        self.rng = random.Random(seed)  # seed нужен, чтобы проверочная выборка была фиксированной
        self.fixed = [self._make() for _ in range(size)] if seed is not None else None

    def _make(self):
        n = self.rng.randint(self.min_len, self.max_len)
        text = "".join(self.rng.choice(self.alphabet) for _ in range(n))
        return to_tensor(Image.open(self.gen.generate(text))), text

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.fixed[i] if self.fixed is not None else self._make()


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return images, labels


# --------------------------------------------------------------------------
# Модель
# --------------------------------------------------------------------------
class CRNN(nn.Module):
    """Свёрточная часть извлекает признаки, LSTM читает их слева направо."""

    def __init__(self, n_classes: int):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

        self.cnn = nn.Sequential(
            block(1, 32), nn.MaxPool2d(2),            # 16 x 64
            block(32, 64), nn.MaxPool2d(2),           # 8 x 32
            block(64, 128), nn.MaxPool2d((2, 1)),     # 4 x 32
            block(128, 128), nn.MaxPool2d((2, 1)),    # 2 x 32
        )
        self.rnn = nn.LSTM(128 * 2, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, n_classes)  # n_classes = символы + 1 (пустой символ CTC)

    def forward(self, x):
        f = self.cnn(x)                                   # B, C, 2, 32
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)    # B, 32 шага, признаки
        out, _ = self.rnn(f)
        return self.fc(out)                               # B, T, классы


# --------------------------------------------------------------------------
# Кодирование / декодирование ответа
# --------------------------------------------------------------------------
def encode(labels, alphabet):
    """Строки -> (склеенные индексы, длины). Индекс 0 зарезервирован под 'blank'."""
    targets, lengths = [], []
    for s in labels:
        targets.extend(alphabet.index(c) + 1 for c in s)
        lengths.append(len(s))
    return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def decode(logits, alphabet):
    """Жадное CTC-декодирование: берём лучший класс на каждом шаге, схлопываем повторы."""
    best = logits.argmax(-1).cpu().numpy()  # B, T
    results = []
    for seq in best:
        chars, prev = [], 0
        for idx in seq:
            if idx != prev and idx != 0:
                chars.append(alphabet[idx - 1])
            prev = idx
        results.append("".join(chars))
    return results


# --------------------------------------------------------------------------
# Сохранение, загрузка, предсказание (используется и captcha_app.py)
# --------------------------------------------------------------------------
def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = CRNN(len(ckpt["alphabet"]) + 1).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt["alphabet"]


@torch.no_grad()
def predict(loaded, pil_image: Image.Image) -> str:
    model, alphabet = loaded
    device = next(model.parameters()).device
    x = to_tensor(pil_image).unsqueeze(0).to(device)
    return decode(model(x), alphabet)[0]


# --------------------------------------------------------------------------
# Обучение
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, alphabet, device):
    model.eval()
    ok = total = 0
    for images, labels in loader:
        preds = decode(model(images.to(device)), alphabet)
        ok += sum(p == t for p, t in zip(preds, labels))
        total += len(labels)
    return ok / max(total, 1)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphabet = args.alphabet

    model = CRNN(len(alphabet) + 1).to(device)
    if args.resume:  # дообучение: берём веса и алфавит из старой модели
        ckpt = torch.load(args.resume, map_location=device)
        alphabet = ckpt["alphabet"]
        model = CRNN(len(alphabet) + 1).to(device)
        model.load_state_dict(ckpt["state"])
        print(f"Продолжаю обучение модели {args.resume}")

    if args.synthetic:
        train_ds = SyntheticDataset(args.epoch_size, alphabet)
        val_ds = SyntheticDataset(500, alphabet, seed=123)
    else:
        full = FolderDataset(args.data, alphabet)
        n_val = max(1, len(full) // 10)
        train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
        print(f"Картинок: {len(full)} (обучение {len(train_ds)}, проверка {len(val_ds)})")
        if len(full) < 500:
            print("Совет: для хорошей точности нужны сотни и тысячи размеченных примеров.")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate)

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            targets, target_lens = encode(labels, alphabet)

            log_probs = model(images).log_softmax(-1).permute(1, 0, 2)  # T, B, C - формат CTC
            input_lens = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)

            loss = ctc(log_probs, targets, input_lens, target_lens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        acc = evaluate(model, val_loader, alphabet, device)
        print(f"Эпоха {epoch}/{args.epochs}  loss={total_loss / len(train_loader):.3f}  точность={acc:.1%}")

        if acc >= best:  # сохраняем лучшую версию
            best = acc
            torch.save({"state": model.state_dict(), "alphabet": alphabet}, args.out)

    print(f"Готово. Лучшая точность на проверке: {best:.1%}. Модель сохранена в {args.out}")


def main():
    p = argparse.ArgumentParser(description="Обучаемый распознаватель капч")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="обучить или дообучить модель")
    t.add_argument("--data", default="dataset", help="папка с размеченными картинками")
    t.add_argument("--synthetic", action="store_true", help="генерировать учебные капчи")
    t.add_argument("--epoch-size", type=int, default=5000, help="картинок за эпоху (synthetic)")
    t.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--resume", help="путь к model.pt для дообучения")
    t.add_argument("--out", default="model.pt")

    r = sub.add_parser("predict", help="распознать картинку")
    r.add_argument("--model", default="model.pt")
    r.add_argument("image")

    args = p.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        print(predict(load_model(args.model), Image.open(args.image)))


if __name__ == "__main__":
    sys.exit(main())