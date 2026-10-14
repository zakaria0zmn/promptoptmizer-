import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

BG0 = "#070707"
BG1 = "#121212"
BORDER = "#2a2a2a"
RED = "#e8273c"
FG = "#f2f2f2"
FG_DIM = "#8f8f8f"

PREVIEW_SIZE = 120
ICON_SIZE = 24
RENDER_SCALE = 4


def draw_line(draw, points, fill, width=1):
    scaled = [(round(x * RENDER_SCALE), round(y * RENDER_SCALE)) for x, y in points]
    draw.line(scaled, fill=fill, width=width * RENDER_SCALE, joint="curve")


def draw_rectangle(draw, bounds, fill):
    scaled = tuple(round(value * RENDER_SCALE) for value in bounds)
    draw.rectangle(scaled, fill=fill)


def draw_hybrid_a(draw):
    draw_line(draw, [(30, 24), (16, 24), (16, 96), (30, 96)], FG, 4)
    draw_line(draw, [(90, 24), (104, 24), (104, 96), (90, 96)], FG, 4)
    draw_rectangle(draw, (28, 36, 58, 42), FG_DIM)
    draw_rectangle(draw, (28, 57, 50, 63), FG_DIM)
    draw_rectangle(draw, (28, 78, 54, 84), FG_DIM)
    draw_rectangle(draw, (64, 36, 70, 84), RED)
    draw_rectangle(draw, (70, 55, 92, 65), RED)


def draw_hybrid_b(draw):
    draw_rectangle(draw, (16, 36, 52, 42), FG_DIM)
    draw_rectangle(draw, (24, 57, 52, 63), FG_DIM)
    draw_rectangle(draw, (16, 78, 52, 84), FG_DIM)
    draw_rectangle(draw, (56, 22, 64, 98), RED)
    draw_rectangle(draw, (68, 54, 92, 66), FG)
    draw_line(draw, [(88, 30), (104, 30), (104, 90), (88, 90)], RED, 4)


def draw_hybrid_c(draw):
    draw_line(draw, [(18, 34), (18, 20), (32, 20)], FG_DIM, 3)
    draw_line(draw, [(18, 86), (18, 100), (32, 100)], FG_DIM, 3)
    draw_line(draw, [(102, 34), (102, 20), (88, 20)], FG_DIM, 3)
    draw_line(draw, [(102, 86), (102, 100), (88, 100)], FG_DIM, 3)
    draw_rectangle(draw, (28, 28, 36, 92), FG_DIM)
    draw_rectangle(draw, (44, 38, 52, 82), FG)
    draw_rectangle(draw, (60, 48, 68, 72), RED)
    draw_rectangle(draw, (76, 55, 92, 65), RED)


def draw_hybrid_d(draw):
    draw_rectangle(draw, (18, 30, 44, 38), FG_DIM)
    draw_rectangle(draw, (18, 46, 52, 54), FG_DIM)
    draw_rectangle(draw, (18, 62, 46, 70), FG_DIM)
    draw_rectangle(draw, (18, 78, 38, 86), FG_DIM)
    draw_line(draw, [(50, 28), (70, 48), (70, 72), (50, 92)], RED, 5)
    draw_rectangle(draw, (76, 54, 98, 66), RED)
    draw_line(draw, [(90, 44), (102, 60), (90, 76)], FG, 4)


def render_logo(draw_fn, size):
    image = Image.new(
        "RGB",
        (PREVIEW_SIZE * RENDER_SCALE, PREVIEW_SIZE * RENDER_SCALE),
        BG0,
    )
    draw_fn(ImageDraw.Draw(image))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main():
    root = tk.Tk()
    root.title("Hybrid Logo Concepts (Anti-aliased Previews)")
    root.configure(bg=BG0, padx=20, pady=20)
    root.attributes("-topmost", True)

    tk.Label(
        root,
        text="CONCEPT 1 GEOMETRY + CONCEPT 2 IDENTITY",
        bg=BG0,
        fg=RED,
        font=("Segoe UI", 11, "bold"),
    ).pack(pady=(0, 14))

    cards = tk.Frame(root, bg=BG0)
    cards.pack()

    concepts = [
        ("1A. Compression Bracket", "[ 3 rough lines -> 1 beam ]", draw_hybrid_a),
        ("1B. The Filter Gate", "Raw lines through a compiler bar", draw_hybrid_b),
        ("1C. Stepped Funnel", "Wide input stepping down to core", draw_hybrid_c),
        ("1D. Signal Focus", "Noise converging into a clear signal", draw_hybrid_d),
    ]
    root.logo_images = []

    for idx, (title, desc, draw_fn) in enumerate(concepts):
        card = tk.Frame(
            cards,
            bg=BG1,
            padx=14,
            pady=12,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        card.grid(row=0, column=idx, padx=8)

        preview_image = ImageTk.PhotoImage(render_logo(draw_fn, PREVIEW_SIZE))
        root.logo_images.append(preview_image)
        preview = tk.Label(
            card,
            image=preview_image,
            bg=BG0,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        preview.pack()

        tk.Label(card, text=title, bg=BG1, fg=FG, font=("Segoe UI", 9, "bold")).pack(
            pady=(10, 2)
        )
        tk.Label(card, text=desc, bg=BG1, fg=FG_DIM, font=("Segoe UI", 8)).pack()

        title_bar = tk.Frame(
            card,
            bg="#202020",
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        title_bar.pack(fill=tk.X, pady=(10, 0))
        icon_image = ImageTk.PhotoImage(render_logo(draw_fn, ICON_SIZE))
        root.logo_images.append(icon_image)
        tk.Label(title_bar, image=icon_image, bg="#202020").pack(
            side=tk.LEFT, padx=4, pady=3
        )
        tk.Label(
            title_bar,
            text=f"{idx + 1}  Preview window",
            bg="#202020",
            fg=FG_DIM,
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT, padx=(2, 6))

    root.mainloop()


if __name__ == "__main__":
    main()
