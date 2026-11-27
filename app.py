import io
import os
import csv
from datetime import datetime
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

# ==========================================
# 1. ページ設定・アプリケーション設定
# ==========================================
st.set_page_config(
    page_title="キュウリ葉面積解析アプリ",
    page_icon="🥒",
    layout="wide",
)

class AppConfig:
    DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
    DEVICE = -1  # CPU強制指定

    # デフォルト設定
    WIDTH_RANGE = (10, 90)   # (左(%), 右(%))
    HEIGHT_RANGE = (0, 100)  # (上(%), 下(%))
    DEPTH_THRESH = 200
    HSV_H = (30, 90)
    HSV_S = (10, 255)
    HSV_V = (0, 255)

    COLORS = {
        "MASK_BLUE": (255, 100, 0),    # 葉以外の背景オーバーレイ青色 (BGR)
        "OUTSIDE_DARK": (30, 30, 30),  # 解析範囲外の暗色オーバーレイ (BGR)
        "OVERLAY_ALPHA": 0.5,
        "OUTSIDE_ALPHA": 0.65,
    }

    SAMPLE_IMAGES = [
        "sample/IMG_3396.jpg",
    ]

# ==========================================
# 2. 推論エンジン (キャッシュ対応 - CPU)
# ==========================================
@st.cache_resource(show_spinner="深度推定モデルをロード中 (初回のみ時間がかかります)...")
def load_depth_pipeline():
    return pipeline(
        "depth-estimation",
        model=AppConfig.DEPTH_MODEL_ID,
        device=AppConfig.DEVICE,
    )

@st.cache_data(show_spinner="深度マップを推定中...")
def estimate_depth(img_bytes: bytes) -> np.ndarray:
    depth_pipe = load_depth_pipeline()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)
    h, w = img_rgb.shape[:2]

    depth_out = depth_pipe(pil_img)
    raw = depth_out["predicted_depth"].squeeze().detach().cpu().numpy()
    norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(norm, (w, h), interpolation=cv2.INTER_LANCZOS4)

# ==========================================
# 3. 葉面積解析メインロジック
# ==========================================
def process_leaf_area(img_bytes: bytes, image_name: str, l: int, r: int, auto_y: bool, t: int, b: int, dt: int, h_range, s_range, v_range):
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(pil_img)
    base_name = os.path.splitext(os.path.basename(image_name))[0]

    h, w = img.shape[:2]
    x1, x2 = int(w * l / 100), int(w * r / 100)

    # 深度マップの取得（キャッシュ済みなら即座に返却）
    depth = estimate_depth(img_bytes)

    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    c_depth = depth[:, x1:x2]
    c_bgr = canvas[:, x1:x2]

    # 深度によるマスク
    mask_d = cv2.threshold(c_depth, 255 - dt, 255, cv2.THRESH_BINARY)[1]

    # HSVカラーによるマスク（緑色の抽出）
    h1, h2 = h_range
    s1, s2 = s_range
    v1, v2 = v_range
    hsv = cv2.cvtColor(cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2HSV)
    mask_c = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([h2, s2, v2]))

    # 深度マスクとカラーマスクの統合
    leaf_mask = cv2.bitwise_and(mask_c, mask_d)

    # 上下幅の決定（自動検出 or 手動調整）
    if auto_y:
        ys, _ = np.where(leaf_mask == 255)
        has_target = len(ys) > 0
        if has_target:
            y_min = int(ys.min())
            y_max = int(ys.max())
            t_out = max(0, min(99, int(round(y_min / h * 100))))
            b_out = max(t_out + 1, min(100, int(round((y_max + 1) / h * 100))))
            y_mode_label = "自動検出"
        else:
            y_min, y_max = 0, h - 1
            t_out, b_out = t, b
            y_mode_label = "自動検出(葉なし)"
    else:
        y_min = int(h * t / 100)
        y_max = max(y_min, int(h * b / 100) - 1)
        has_target = (y_max >= y_min) and (x2 > x1)
        t_out, b_out = int(t), int(b)
        y_mode_label = "手動調整"

    if has_target:
        roi_mask = leaf_mask[y_min:y_max + 1, :]
        pixel_count = int(cv2.countNonZero(roi_mask))
        eff_height = y_max - y_min + 1
        total_pixels = eff_height * (x2 - x1)
        leaf_pct = (pixel_count / total_pixels) * 100
        y_info_text = f"上下({y_mode_label}): {y_min}〜{y_max} px（高さ {eff_height} px / {t_out}%〜{b_out}%）"

        # 解析対象範囲（x1:x2, y_min:y_max+1）内のオーバーレイ作成
        roi_bgr = canvas[y_min:y_max + 1, x1:x2]
        blue_bg = np.full_like(roi_bgr, AppConfig.COLORS["MASK_BLUE"])
        canvas[y_min:y_max + 1, x1:x2] = np.where(
            roi_mask[:, :, None] == 255,
            roi_bgr,
            cv2.addWeighted(
                roi_bgr,
                1 - AppConfig.COLORS["OVERLAY_ALPHA"],
                blue_bg,
                AppConfig.COLORS["OVERLAY_ALPHA"],
                0,
            ),
        )

        # 分母範囲外（左右の除外領域 + 上下の範囲外）を暗く表示
        outside_mask = np.ones((h, w), dtype=bool)
        outside_mask[y_min:y_max + 1, x1:x2] = False
        dark_bg = np.full_like(canvas, AppConfig.COLORS["OUTSIDE_DARK"])
        darkened = cv2.addWeighted(
            canvas, 1 - AppConfig.COLORS["OUTSIDE_ALPHA"], dark_bg, AppConfig.COLORS["OUTSIDE_ALPHA"], 0
        )
        canvas[outside_mask] = darkened[outside_mask]

        # 分母となる囲み領域の枠線を描画
        cv2.rectangle(canvas, (x1, y_min), (x2 - 1, y_max), (255, 255, 255), 2)
    else:
        pixel_count = 0
        total_pixels = leaf_mask.size
        leaf_pct = 0.0
        y_info_text = "上下(自動検出): 葉が検出されませんでした"
        dark_bg = np.full_like(canvas, AppConfig.COLORS["OUTSIDE_DARK"])
        canvas = cv2.addWeighted(
            canvas, 1 - AppConfig.COLORS["OUTSIDE_ALPHA"], dark_bg, AppConfig.COLORS["OUTSIDE_ALPHA"], 0
        )

    res_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    now = datetime.now()
    result_data = {
        "base_name": base_name,
        "date_str": now.strftime("%Y%m%d"),
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": os.path.basename(image_name),
        "leaf_area_pct": round(leaf_pct, 2),
        "extracted_pixels": pixel_count,
        "total_pixels": total_pixels,
        "roi_range": f"L:{l}%, R:{r}% / T:{t_out}%, B:{b_out}% ({y_mode_label}, Y:{y_min}-{y_max}px)",
        "l": l,
        "r": r,
        "t_out": t_out,
        "b_out": b_out,
        "y_info_text": y_info_text,
        "depth_thresh": dt,
        "hsv_range": f"H:{h1}-{h2}, S:{s1}-{s2}, V:{v1}-{v2}",
    }

    return res_img, result_data

def build_csv_bytes(result_data: dict) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "解析日時", "画像ファイル名", "葉面積率(%)",
        "抽出ピクセル数(px)", "囲み範囲総ピクセル数(px)",
        "解析範囲", "深度しきい値", "HSVフィルタ範囲",
    ])
    writer.writerow([
        result_data["timestamp"],
        result_data["image_name"],
        result_data["leaf_area_pct"],
        result_data["extracted_pixels"],
        result_data["total_pixels"],
        result_data["roi_range"],
        result_data["depth_thresh"],
        result_data["hsv_range"],
    ])
    return buf.getvalue().encode("utf-8-sig")

# ==========================================
# 4. セッション状態初期化 & コールバック
# ==========================================
if "auto_y" not in st.session_state:
    st.session_state["auto_y"] = True
if "y_range_slider" not in st.session_state:
    st.session_state["y_range_slider"] = AppConfig.HEIGHT_RANGE
if "pending_y_range" in st.session_state:
    st.session_state["y_range_slider"] = st.session_state.pop("pending_y_range")

def on_y_slider_change():
    st.session_state["auto_y"] = False

# ==========================================
# 5. UI構築 (Streamlit Online)
# ==========================================
st.title("🥒 キュウリ葉面積率算出アプリ")
st.caption(
    "横幅を調整して解析すると上下の葉位置を自動検出します。必要に応じて詳細設定から上下幅を手動で微調整できます。"
)

col_in, col_out = st.columns([2, 3], gap="large")

with col_in:
    available_samples = [p for p in AppConfig.SAMPLE_IMAGES if os.path.exists(p)]

    img_bytes = None
    image_name = None

    if available_samples:
        input_mode = st.radio(
            "画像入力方法",
            ["画像をアップロード", "サンプル画像を使用"],
            horizontal=True,
        )
    else:
        input_mode = "画像をアップロード"

    if input_mode == "画像をアップロード":
        uploaded_file = st.file_uploader(
            "撮影画像をアップロード",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            img_bytes = uploaded_file.getvalue()
            image_name = uploaded_file.name
            st.image(img_bytes, caption=f"入力画像: {image_name}", use_container_width=True)
    else:
        selected_sample = st.selectbox("サンプル画像を選択", available_samples)
        with open(selected_sample, "rb") as f:
            img_bytes = f.read()
        image_name = os.path.basename(selected_sample)
        st.image(img_bytes, caption=f"サンプル画像: {image_name}", use_container_width=True)

    # 左右幅の調節バー（アコーディオン外に直接配置）
    width_range = st.slider(
        "📐 解析する横幅の範囲 (%)",
        min_value=0,
        max_value=100,
        value=AppConfig.WIDTH_RANGE,
        step=1,
    )

    # 合体した1つのアコーディオン：上下幅・深度しきい値・HSVカラーフィルタ
    with st.expander("⚙️ 詳細設定（上下幅・深度・HSV）", expanded=False):
        st.markdown("**📐 上下幅設定**")
        auto_y = st.checkbox(
            "上下幅を自動検出する（葉の上下端に合わせる）",
            key="auto_y",
        )
        y_range = st.slider(
            "上下幅の範囲 (%)",
            min_value=0,
            max_value=100,
            step=1,
            key="y_range_slider",
            on_change=on_y_slider_change,
        )

        st.markdown("**📏 深度しきい値設定**")
        dt = st.slider(
            "深度しきい値",
            min_value=0,
            max_value=255,
            value=AppConfig.DEPTH_THRESH,
            step=1,
        )

        st.markdown("**🎨 HSVカラーフィルタ設定**")
        h_range = st.slider("H (色相) 範囲", 0, 179, AppConfig.HSV_H, step=1)
        s_range = st.slider("S (彩度) 範囲", 0, 255, AppConfig.HSV_S, step=1)
        v_range = st.slider("V (明度) 範囲", 0, 255, AppConfig.HSV_V, step=1)

    run_btn = st.button("🚀 葉面積を計算する", type="primary", use_container_width=True)

if run_btn:
    if img_bytes is None:
        st.session_state["leaf_error"] = "画像を選択またはアップロードしてください。"
        st.session_state.pop("leaf_result", None)
    elif width_range[0] >= width_range[1]:
        st.session_state["leaf_error"] = "横幅の左端と右端は異なる値（左 < 右）に設定してください。"
        st.session_state.pop("leaf_result", None)
    elif not auto_y and y_range[0] >= y_range[1]:
        st.session_state["leaf_error"] = "上下幅の上端と下端は異なる値（上 < 下）に設定してください。"
        st.session_state.pop("leaf_result", None)
    else:
        st.session_state.pop("leaf_error", None)
        res_img, result_data = process_leaf_area(
            img_bytes=img_bytes,
            image_name=image_name,
            l=width_range[0],
            r=width_range[1],
            auto_y=auto_y,
            t=y_range[0],
            b=y_range[1],
            dt=dt,
            h_range=h_range,
            s_range=s_range,
            v_range=v_range,
        )
        st.session_state["leaf_result"] = {
            "res_img": res_img,
            "data": result_data,
        }
        # 自動検出した上下幅をスライダーに反映して再描画
        if auto_y and result_data["extracted_pixels"] > 0:
            st.session_state["pending_y_range"] = (result_data["t_out"], result_data["b_out"])
            st.rerun()

with col_out:
    if "leaf_error" in st.session_state:
        st.error(st.session_state["leaf_error"])
    elif "leaf_result" in st.session_state:
        res_img = st.session_state["leaf_result"]["res_img"]
        d = st.session_state["leaf_result"]["data"]

        summary_html = f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f4f9f4; border: 2px solid #2e8b57; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.08); margin-bottom: 16px;">
            <h3 style="margin: 0 0 10px 0; color: #2e8b57; font-size: 1.35rem;">🌿 葉面積解析結果 ({d['base_name']})</h3>
            <div style="margin: 12px 0; font-size: 2.3rem; font-weight: bold; color: #222;">
                葉面積率: <span style="color: #2e8b57; font-size: 2.8rem;">{d['leaf_area_pct']:.2f}</span> <span style="font-size: 1.6rem;">%</span>
            </div>
            <div style="font-size: 1.05rem; color: #555;">
                抽出ピクセル数: <b>{d['extracted_pixels']:,}</b> / {d['total_pixels']:,} px（囲み範囲内）
            </div>
            <div style="font-size: 0.9rem; color: #666; margin-top: 6px;">
                解析横幅: {d['l']}% 〜 {d['r']}% ／ {d['y_info_text']}
            </div>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)
        st.image(res_img, caption="解析結果オーバーレイ", use_container_width=True)

        csv_bytes = build_csv_bytes(d)
        st.download_button(
            label="📥 結果をCSVでダウンロード",
            data=csv_bytes,
            file_name=f"{d['base_name']}_{d['date_str']}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("左側のパネルから画像を選択し、「🚀 葉面積を計算する」を押してください。")
