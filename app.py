import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Pengaturan Halaman
st.set_page_config(
    page_title="Aplikasi Segmentasi Citra Bola v1.3",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ Aplikasi Segmentasi Citra Bola v1.3")
st.markdown(
    "Aplikasi segmentasi objek bola berbasis *Thresholding* dan **Komparasi Matriks Piksel Menyeluruh**."
)

# ==========================================
# NAVBAR / SIDEBAR KONTROL
# ==========================================
st.sidebar.header("⚙️ Pengaturan & Metode")

method = st.sidebar.selectbox(
    "Pilih Metode Thresholding:",
    [
        "Otsu Binarization",
        "Global Binary (T Konstan)",
        "Global Binary Inverted",
        "Truncate",
        "To-Zero",
        "Adaptive Gaussian",
        "Adaptive Mean",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Slider Parameter Thresholding")

thresh_val = 127
max_val = 255
block_size = 11
c_val = 2

if method in [
    "Global Binary (T Konstan)",
    "Global Binary Inverted",
    "Truncate",
    "To-Zero",
]:
    thresh_val = st.sidebar.slider(
        "Nilai Threshold (T):", min_value=0, max_value=255, value=127, step=1
    )
    max_val = st.sidebar.slider(
        "Nilai Maksimum (Max Value):",
        min_value=0,
        max_value=255,
        value=255,
        step=1,
    )

elif method in ["Adaptive Gaussian", "Adaptive Mean"]:
    block_size_raw = st.sidebar.slider(
        "Block Size (Ganjil > 1):",
        min_value=3,
        max_value=99,
        value=11,
        step=2,
    )
    block_size = (
        block_size_raw if block_size_raw % 2 != 0 else block_size_raw + 1
    )
    c_val = st.sidebar.slider(
        "Konstanta C:", min_value=-20, max_value=20, value=2, step=1
    )

elif method == "Otsu Binarization":
    st.sidebar.info(
        "ℹ️ Otsu menghitung nilai threshold secara otomatis dari histogram."
    )

# ==========================================
# UNGGAN GAMBAR
# ==========================================
uploaded_file = st.file_uploader(
    "Unggah Gambar Bola Anda (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Konversi ke Grayscale
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Pra-pemrosesan
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ==========================================
    # LOGIKA THRESHOLDING
    # ==========================================
    computed_thresh = None

    if method == "Otsu Binarization":
        computed_thresh, binary_mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "Global Binary (T Konstan)":
        _, binary_mask = cv2.threshold(
            blurred, thresh_val, max_val, cv2.THRESH_BINARY
        )
    elif method == "Global Binary Inverted":
        _, binary_mask = cv2.threshold(
            blurred, thresh_val, max_val, cv2.THRESH_BINARY_INV
        )
    elif method == "Truncate":
        _, binary_mask = cv2.threshold(
            blurred, thresh_val, max_val, cv2.THRESH_TRUNC
        )
    elif method == "To-Zero":
        _, binary_mask = cv2.threshold(
            blurred, thresh_val, max_val, cv2.THRESH_TOZERO
        )
    elif method == "Adaptive Gaussian":
        binary_mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_val,
        )
    elif method == "Adaptive Mean":
        binary_mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_val,
        )

    # Hasil Segmentasi
    if len(img_array.shape) == 3:
        segmented_img = cv2.bitwise_and(
            img_array, img_array, mask=binary_mask
        )
    else:
        segmented_img = cv2.bitwise_and(gray, gray, mask=binary_mask)

    # ==========================================
    # VISUALISASI GAMBAR
    # ==========================================
    st.markdown("---")
    st.subheader(f"📌 Hasil Segmentasi — {method}")

    if computed_thresh is not None:
        st.success(
            f"**Nilai Threshold Otomatis Terhitung (Otsu):** {computed_thresh:.0f}"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_array, caption="1. Citra Asli", use_column_width=True)
    with col2:
        st.image(
            binary_mask,
            caption=f"2. Masker Biner ({method})",
            use_column_width=True,
            clamp=True,
        )
    with col3:
        st.image(
            segmented_img,
            caption="3. Hasil Segmentasi Bola",
            use_column_width=True,
        )

    # ==========================================
    # VERSI 1.3: KOMPARASI MATRIKS SELURUH CITRA
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Komparasi Matriks Seluruh Citra (v1.3)")

    h, w = gray.shape
    st.write(f"**Ukuran Citra Asli:** {w} × {h} Piksel (Total {w*h:,} Piksel)")

    mode_matriks = st.radio(
        "Pilih Mode Tampilan Matriks:",
        [
            "Tampilkan Matriks Keseluruhan (Downsampled / Dikecilkan)",
            "Tampilkan Seluruh Matriks Asli (Full Resolution)",
        ],
    )

    if (
        mode_matriks
        == "Tampilkan Matriks Keseluruhan (Downsampled / Dikecilkan)"
    ):
        dim = st.slider(
            "Resolusi Matriks Tampilan (Piksel x Piksel):",
            min_value=10,
            max_value=64,
            value=25,
            step=5,
        )

        # Resize citra khusus untuk tampilan matriks tanpa merusak proses segmentasi utama
        gray_resized = cv2.resize(
            gray, (dim, dim), interpolation=cv2.INTER_AREA
        )
        binary_resized = cv2.resize(
            binary_mask, (dim, dim), interpolation=cv2.INTER_NEAREST
        )

        df_gray_full = pd.DataFrame(
            gray_resized,
            columns=[f"X{i}" for i in range(dim)],
            index=[f"Y{j}" for j in range(dim)],
        )
        df_binary_full = pd.DataFrame(
            binary_resized,
            columns=[f"X{i}" for i in range(dim)],
            index=[f"Y{j}" for j in range(dim)],
        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**Matriks Citra Asli (Grayscale {dim}x{dim})**")
            st.dataframe(df_gray_full, use_container_width=True)

        with col_m2:
            st.markdown(
                f"**Matriks Hasil Segmentasi / Thresholding ({dim}x{dim})**"
            )
            st.dataframe(df_binary_full, use_container_width=True)

    else:
        st.warning(
            "⚠️ Memuat seluruh matriks asli ($w \\times h$) dapat memakan waktu sesuai spesifikasi komputer Anda."
        )

        df_gray_orig = pd.DataFrame(
            gray,
            columns=[f"X{i}" for i in range(w)],
            index=[f"Y{j}" for j in range(h)],
        )
        df_binary_orig = pd.DataFrame(
            binary_mask,
            columns=[f"X{i}" for i in range(w)],
            index=[f"Y{j}" for j in range(h)],
        )

        tab1, tab2 = st.tabs(
            [
                "📋 Matriks Citra Asli (Grayscale)",
                f"📋 Matriks Hasil Segmentasi ({method})",
            ]
        )

        with tab1:
            st.dataframe(df_gray_orig, use_container_width=True)
            # Ekspor CSV
            csv_gray = df_gray_orig.to_csv().encode("utf-8")
            st.download_button(
                "📥 Unduh Matriks Asli (CSV)",
                data=csv_gray,
                file_name="matriks_citra_asli.csv",
                mime="text/csv",
            )

        with tab2:
            st.dataframe(df_binary_orig, use_container_width=True)
            # Ekspor CSV
            csv_binary = df_binary_orig.to_csv().encode("utf-8")
            st.download_button(
                "📥 Unduh Matriks Segmentasi (CSV)",
                data=csv_binary,
                file_name=f"matriks_segmentasi_{method}.csv",
                mime="text/csv",
            )

else:
    st.info("💡 Silakan unggah gambar bola untuk memulai proses segmentasi.")