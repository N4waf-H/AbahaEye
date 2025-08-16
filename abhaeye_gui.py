# abhaeye_gui_ar.py
import os
import io
import tempfile
import threading
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# محلّل الشدة (إصدار العربية)
import openai_severity  # تأكد أن الملف العربي موجود بنفس المجلد

MODEL_ID = "hilmantm/detr-traffic-accident-detection"
ACCIDENT_KEYWORDS = ("accident", "crash", "collision")  # تسميات الموديل إنجليزيّة


@dataclass
class DetectionResult:
    accident_present: bool
    max_accident_score: float
    num_accident_detections: int
    vis_image: Image.Image


def is_accident_label(name: str) -> bool:
    n = name.lower().strip()
    return any(k in n for k in ACCIDENT_KEYWORDS)


def load_model(device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    model.to(device).eval()
    return processor, model, device


def label_name_map(model) -> Dict[int, str]:
    id2label = getattr(model.config, "id2label", None)
    if id2label is None or len(id2label) == 0:
        return {}
    return {int(k): v for k, v in id2label.items()} if isinstance(id2label, dict) else id2label


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    return font.getsize(text)


def postprocess(processor, outputs, pil_img: Image.Image, score_threshold: float = 0.05):
    target_sizes = torch.tensor([pil_img.size[::-1]], device=outputs.logits.device)  # (h, w)
    results = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)[0]
    scores = results["scores"].detach().cpu().numpy()
    labels = results["labels"].detach().cpu().numpy()
    boxes = results["boxes"].detach().cpu().numpy()
    keep = scores >= score_threshold
    return {"scores": scores[keep], "labels": labels[keep], "boxes": boxes[keep]}


def decide_accident(id2label: Dict[int, str], scores: np.ndarray, labels: np.ndarray, threshold: float):
    max_score = 0.0
    count = 0
    for s, lid in zip(scores, labels):
        name = id2label.get(int(lid), str(lid))
        if is_accident_label(name) and s >= threshold:
            count += 1
            max_score = max(max_score, float(s))
    return (count > 0, max_score, count)


def draw_boxes(img: Image.Image, boxes, scores, labels, id2label, accident_threshold: float, decision: bool):
    """
    نرسم صناديق حمراء فقط حول الحوادث (بدون أي نصوص داخل الصورة).
    """
    draw = ImageDraw.Draw(img)

    for box, score, lab in zip(boxes, scores, labels):
        name = id2label.get(int(lab), str(lab))
        # فقط لو كان وسم حادث وتجاوز العتبة
        if not is_accident_label(name) or score < accident_threshold:
            continue

        x0, y0, x1, y1 = map(int, box.tolist())
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

    return img


def load_image_from_path(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def analyze_image(pil_img: Image.Image, processor, model, device, score_filter: float, accident_threshold: float) -> DetectionResult:
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    det = postprocess(processor, outputs, pil_img, score_threshold=score_filter)
    scores = det["scores"]
    labels = det["labels"]
    boxes = det["boxes"]
    id2label = label_name_map(model)

    decision, max_acc, num_acc = decide_accident(id2label, scores, labels, threshold=accident_threshold)

    vis = pil_img.copy()
    vis = draw_boxes(vis, boxes, scores, labels, id2label, accident_threshold, decision)

    return DetectionResult(decision, float(max_acc), int(num_acc), vis)


def resize_to_fit(pil_img: Image.Image, max_w=960, max_h=720) -> Image.Image:
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_img


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("عين أبها  لرصد الحوادث (DETR) + تقييم الشدة (GPT-4o)")
        self.geometry("1500x860")

        self.processor = None
        self.model = None
        self.device = None
        self.current_image = None
        self.tk_image = None

        self._build_ui()
        self._load_model_async()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        # أدوات الإدخال
        self.btn_open = ttk.Button(top, text="فتح صورة…", command=self.choose_image, state=tk.DISABLED)
        self.btn_open.pack(side=tk.RIGHT, padx=5)  # يمين لأن الواجهة عربية

        self.url_var = tk.StringVar()
        url_entry = ttk.Entry(top, textvariable=self.url_var, width=60, justify="right")
        url_entry.pack(side=tk.RIGHT, padx=5)
        self.btn_url = ttk.Button(top, text="تحميل من رابط", command=self.load_url, state=tk.DISABLED)
        self.btn_url.pack(side=tk.RIGHT, padx=5)

        ttk.Label(top, text="حدّ العتبة للحادث").pack(side=tk.RIGHT, padx=(15, 5))
        self.thresh_var = tk.DoubleVar(value=0.55)
        self.thresh = ttk.Scale(top, from_=0.1, to=0.9, orient=tk.HORIZONTAL, variable=self.thresh_var, length=150)
        self.thresh.pack(side=tk.RIGHT, padx=5)
        self.thresh_value_lbl = ttk.Label(top, text="0.55")
        self.thresh_value_lbl.pack(side=tk.RIGHT, padx=(2, 8))
        self.thresh_var.trace_add("write", lambda *_: self.thresh_value_lbl.config(text=f"{self.thresh_var.get():.2f}"))

        self.btn_analyse = ttk.Button(top, text="حلّل", command=self.run_analysis, state=tk.DISABLED)
        self.btn_analyse.pack(side=tk.RIGHT, padx=5)

        # اللوحة الرئيسية
        main = ttk.Frame(self, padding=(10, 0, 10, 10))
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # لوحة المعلومات الجانبية (يمين)
        side = ttk.Frame(main, padding=10, width=450)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        ttk.Label(side, text="شدة الحادث (GPT-4o)", font=("Segoe UI", 16, "bold"), anchor="e", justify="right").pack(anchor="e", pady=(0, 6))
        self.sev_val = ttk.Label(side, text="-", font=("Segoe UI", 14), anchor="e", justify="right")
        self.sev_val.pack(anchor="e", pady=(0, 10), fill=tk.X)

        ttk.Label(side, text="الجهات المقترحة", font=("Segoe UI", 16, "bold"), anchor="e", justify="right").pack(anchor="e", pady=(6, 6))
        self.dispatch_val = ttk.Label(side, text="-", wraplength=360, justify="right", anchor="e", font=("Segoe UI", 14))
        self.dispatch_val.pack(anchor="e", pady=(0, 10), fill=tk.X)

        ttk.Label(side, text="السبب", font=("Segoe UI", 16, "bold"), anchor="e", justify="right").pack(anchor="e", pady=(6, 6))
        self.reason_text = tk.Text(side, height=10, wrap="word", font=("Segoe UI", 14))
        self.reason_text.pack(fill=tk.BOTH, expand=False)
        self.reason_text.tag_configure("rtl", justify="right")

        
        image_panel = ttk.Frame(main, padding=(0, 0, 10, 0))
        image_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # إطار للشريط العلوي
        self.banner_frame = tk.Frame(image_panel, height=60, bg="gray")
        self.banner_frame.pack(side=tk.TOP, fill=tk.X)

        self.acc_label = tk.Label(
            self.banner_frame,
            text="—",
            font=("Segoe UI", 22, "bold"),
            anchor="center",
            justify="center",
            bg="gray",
            fg="white"
        )
        self.acc_label.pack(fill=tk.BOTH, expand=True)

        # تصغير الكانفس
        self.canvas = tk.Canvas(image_panel, bg="#222", width=900, height=600)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # شريط الحالة أسفل
        self.status = ttk.Label(self, text="جاري تحميل النموذج…", padding=8, anchor="e", justify="right")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # زر الإدخال السريع
        self.bind("<Return>", lambda _: self.run_analysis())

    def _load_model_async(self):
        def _worker():
            try:
                self.processor, self.model, self.device = load_model()
                self.status.config(text=f"النموذج جاهز على {self.device}. حمِّل صورة أو الصق رابطًا ثم اضغط «حلّل».")
                self.btn_open.config(state=tk.NORMAL)
                self.btn_url.config(state=tk.NORMAL)
                self.btn_analyse.config(state=tk.NORMAL)
            except Exception as e:
                self.status.config(text="حدث خطأ أثناء تحميل النموذج.")
                messagebox.showerror("خطأ في التحميل", str(e))
        threading.Thread(target=_worker, daemon=True).start()

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="اختر صورة",
            filetypes=[("ملفات الصور", "*.jpg *.jpeg *.png")]
        )
        if not path:
            return
        try:
            img = load_image_from_path(path)
        except Exception as e:
            messagebox.showerror("خطأ في الصورة", str(e))
            return
        self.current_image = img
        self._show_image(img)
        self._clear_gpt_panel()
        self.acc_label.config(text="—", foreground="black")
        self.status.config(text=f"تم التحميل: {os.path.basename(path)} | اضغط «حلّل» للتشغيل")

    def load_url(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showinfo("الرابط", "الرجاء لصق رابط صورة أولًا.")
            return
        self.status.config(text="جاري تنزيل الصورة…")
        self.update_idletasks()
        try:
            img = load_image_from_url(url)
        except Exception as e:
            self.status.config(text="فشل تنزيل الرابط.")
            messagebox.showerror("خطأ في الرابط", str(e))
            return
        self.current_image = img
        self._show_image(img)
        self._clear_gpt_panel()
        self.acc_label.config(text="—", foreground="black")
        self.status.config(text="تم التحميل من الرابط | اضغط «حلّل» للتشغيل")

    def _clear_gpt_panel(self):
        self.sev_val.config(text="-")
        self.dispatch_val.config(text="-")
        self.reason_text.delete("1.0", "end")

    def run_analysis(self):
        if self.current_image is None:
            messagebox.showinfo("لا توجد صورة", "حمِّل صورة أولًا.")
            return
        if self.model is None:
            messagebox.showinfo("النموذج", "لا يزال النموذج يُحمَّل. الرجاء الانتظار.")
            return

        img = self.current_image.copy()
        thresh = float(self.thresh_var.get())
        self.status.config(text="جاري التحليل…")
        self.update_idletasks()

        def _worker():
            try:
                # 1) كشف الحادث عبر DETR
                det_res = analyze_image(
                    pil_img=img, processor=self.processor, model=self.model, device=self.device,
                    score_filter=0.05, accident_threshold=thresh
                )

                vis_resized = resize_to_fit(det_res.vis_image)
                decision_text = "حادث" if det_res.accident_present else "لا يوجد حادث"
                if det_res.accident_present:
                    self.banner_frame.config(bg="#C00000")
                    self.acc_label.config(
                        text=decision_text,
                        bg="#C00000",
                        fg="white"
                    )
                else:
                    self.banner_frame.config(bg="#0A8A0A")
                    self.acc_label.config(
                        text=decision_text,
                        bg="#0A8A0A",
                        fg="white"
                    )

                if det_res.accident_present:
                    # 2) حفظ مؤقت للصورة المعدلة (مع الصناديق) وتمريرها لـ LLM
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                        det_res.vis_image.save(tmp_path, format="PNG")

                    try:
                        sev_raw = openai_severity.analyze_image(tmp_path)
                        sev = openai_severity.format_for_gui(sev_raw)  # استخدام الفاصل العربي "، "
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                    # 3) تحديث اللوحة الجانبية
                    severity = sev.get("severity_level", "غير معروف")
                    dispatch = sev.get("recommended_dispatch_str", "-")
                    reasoning = sev.get("reasoning_short", "")

                    self.sev_val.config(text=severity)
                    self.dispatch_val.config(text=dispatch)
                    self.reason_text.delete("1.0", "end")
                    self.reason_text.insert("1.0", reasoning)
                    self.reason_text.tag_add("rtl", "1.0", "end")

                    self.status.config(
                        text=f"حادث | أعلى درجة ثقة: {det_res.max_accident_score:.2f} | عدد الكشوف: {det_res.num_accident_detections} | العتبة: {thresh:.2f} | الشدة: {severity}"
                    )
                else:
                    self._clear_gpt_panel()
                    self.status.config(
                        text=f"لا يوجد حادث | أعلى درجة ثقة: {det_res.max_accident_score:.2f} | عدد الكشوف: {det_res.num_accident_detections} | العتبة: {thresh:.2f}"
                    )

            except Exception as e:
                self.status.config(text="فشل التحليل.")
                messagebox.showerror("خطأ في التحليل", str(e))

        threading.Thread(target=_worker, daemon=True).start()


    def _show_image(self, pil_img: Image.Image):
        pil_img = resize_to_fit(pil_img)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        W, H = self.canvas.winfo_width(), self.canvas.winfo_height()
        if W < 10 or H < 10:
            W, H = 1000, 700
        x = (W - self.tk_image.width()) // 2
        y = (H - self.tk_image.height()) // 2
        self.canvas.create_image(max(0, x), max(0, y), anchor="nw", image=self.tk_image)


if __name__ == "__main__":
    # لمنع عدم الوضوح في ويندوز
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = App()
    app.mainloop()
