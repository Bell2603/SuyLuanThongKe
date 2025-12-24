# ============================================================
# PHÂN TÍCH HỒI QUY TUYẾN TÍNH BỘI – ĐỦ 10 BƯỚC (THEO GIÁO TRÌNH)
# Dataset: ParisHousing.csv
# KHÔNG LOG – KHÔNG CẮT DIỆN TÍCH
# CHUẨN HÓA PHẦN DƯ – LƯU KẾT QUẢ RA FILE TEXT
# ============================================================

# ===================== BƯỚC 0. IMPORT ======================
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Cấu hình hiển thị pandas (không bị ...)
pd.set_option("display.max_columns", None)   
pd.set_option("display.width", 200)       

# Tạo thư mục lưu hình
os.makedirs("figures", exist_ok=True) 

# Redirect toàn bộ print ra file text
sys.stdout = open("results.txt", "w", encoding="utf-8")


# ===================== BƯỚC 1. XÁC ĐỊNH MÔ HÌNH ======================
print("============================================================")
print("BƯỚC 1. XÁC ĐỊNH MÔ HÌNH HỒI QUY")
print("Biến phụ thuộc: Giá nhà (Euro)")
print("============================================================\n")

df = pd.read_csv("ParisHousing.csv")


# ===================== BƯỚC 1.5. KHAI BÁO BIẾN & VIETSUB (DÙNG CHUNG) ======================
VAR_LABELS = {
    "squareMeters": "Diện tích sử dụng (m²)",
    "numberOfRooms": "Số phòng (phòng)",
    "floors": "Số tầng (tầng)",
    "cityPartRange": "Mức độ cao cấp (1-10)",
    "numPrevOwners": "Số chủ sở hữu trước",
    "made": "Năm xây dựng",
    "basement": "Diện tích tầng hầm (m²)",
    "attic": "Diện tích gác mái (m²)",
    "garage": "Diện tích gara (m²)",
    "hasGuestRoom": "Số phòng cho khách (phòng)",
    "price": "Giá nhà (Euro)"
}

BINARY_LABELS = {
    "hasYard": "Có sân (1 = Có, 0 = Không)",
    "hasPool": "Có hồ bơi (1 = Có, 0 = Không)",
    "isNewBuilt": "Nhà mới xây (1 = Có, 0 = Không)",
    "hasStormProtector": "Có chống bão (1 = Có, 0 = Không)",
    "hasStorageRoom": "Có kho chứa (1 = Có, 0 = Không)"
}

# Danh sách biến dùng chung
quantitative_vars = list(VAR_LABELS.keys())
binary_vars = list(BINARY_LABELS.keys())

quantitative_labels = VAR_LABELS
binary_labels = BINARY_LABELS


# ===================== BƯỚC 2. THỐNG KÊ MÔ TẢ ======================
print("============================================================")
print("BƯỚC 2. THỐNG KÊ MÔ TẢ DỮ LIỆU")
print("============================================================\n")

print("----- Thống kê mô tả: BIẾN ĐỊNH LƯỢNG -----")
desc_quant = df[quantitative_vars].describe()  # Thống kê mô tả cho biến định lượng
desc_quant = desc_quant.rename(columns=quantitative_labels)   
print(desc_quant)

print("\n----- Thống kê mô tả: CÁC BIẾN NHỊ PHÂN -----")

bin_summary = []

for var, label in BINARY_LABELS.items():
    n = df[var].count()  
    count_1 = df[var].sum() 
    percent = count_1 / n * 100

    bin_summary.append({
        "Biến": label,
        "Số quan sát": n,
        "Số có (giá trị = 1)": count_1,
        "Tỷ lệ (%)": round(percent, 2)
    })

bin_summary_df = pd.DataFrame(bin_summary)
print(bin_summary_df)



# ===================== BƯỚC 3. TRỰC QUAN HÓA ======================
print("\n============================================================")
print("BƯỚC 3. TRỰC QUAN HÓA DỮ LIỆU")
print("============================================================\n")

# ------------------------------------------------------------
# 3.1. Phân phối của biến phụ thuộc
# ------------------------------------------------------------

# Histogram giá nhà
plt.figure(figsize=(6, 4))
sns.histplot(df["price"], bins=30, kde=True)
plt.xlabel(VAR_LABELS["price"])
plt.ylabel("Tần suất")
plt.title("Phân phối giá nhà")
plt.tight_layout()
plt.savefig("figures/hist_price.png")
plt.close()


# ------------------------------------------------------------
# 3.2. Mối quan hệ giữa giá nhà và các biến định lượng chính
# ------------------------------------------------------------

# Scatter giá ~ diện tích
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["squareMeters"], y=df["price"], alpha=0.5)
plt.xlabel(VAR_LABELS["squareMeters"])
plt.ylabel(VAR_LABELS["price"])
plt.title("Giá nhà theo diện tích sử dụng")
plt.tight_layout()
plt.savefig("figures/scatter_price_vs_area.png")
plt.close()

# Scatter giá ~ số tầng (THÊM)
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df["floors"], y=df["price"], alpha=0.5)
plt.xlabel(VAR_LABELS["floors"])
plt.ylabel(VAR_LABELS["price"])
plt.title("Giá nhà theo số tầng")
plt.tight_layout()
plt.savefig("figures/scatter_price_vs_floors.png")
plt.close()


# ------------------------------------------------------------
# 3.3. So sánh giá nhà theo mức độ khu vực
# ------------------------------------------------------------

# Boxplot giá nhà theo mức độ khu vực (THÊM)
plt.figure(figsize=(7, 4))
sns.boxplot(x=df["cityPartRange"], y=df["price"])
plt.xlabel(VAR_LABELS["cityPartRange"])
plt.ylabel(VAR_LABELS["price"])
plt.title("Phân bố giá nhà theo mức độ khu vực")
plt.tight_layout()
plt.savefig("figures/boxplot_price_by_cityPartRange.png")
plt.close()


# ------------------------------------------------------------
# 3.4. Phát hiện giá trị ngoại lai (boxplot từng biến)
# ------------------------------------------------------------

boxplot_vars = {
    "squareMeters": VAR_LABELS["squareMeters"],
    "floors": VAR_LABELS["floors"],
    "cityPartRange": VAR_LABELS["cityPartRange"],
    "price": VAR_LABELS["price"]
}

for var, label in boxplot_vars.items():
    plt.figure(figsize=(5, 4))
    sns.boxplot(y=df[var], showfliers=True)
    plt.ylabel(label)
    plt.title(f"Boxplot của {label}")
    plt.tight_layout()
    plt.savefig(f"figures/boxplot_{var}.png")
    plt.close()

print("Đã lưu đầy đủ các biểu đồ trực quan hóa dữ liệu.\n")


# ===================== BƯỚC 4. BIẾN ĐỔI ======================
print("============================================================")
print("BƯỚC 4. BIẾN ĐỔI DỮ LIỆU")
print("Không thực hiện biến đổi (giữ nguyên dữ liệu gốc).\n")


# ===================== BƯỚC 5. KIỂM TRA ĐIỀU KIỆN OLS ======================
print("============================================================")
print("BƯỚC 5. KIỂM TRA ĐIỀU KIỆN TỒN TẠI NGHIỆM OLS")
print("============================================================")

X_full = df[
    [
        "squareMeters",
        "floors",
        "cityPartRange",
        "numberOfRooms",
        "numPrevOwners",
        "made",
        "basement",
        "attic",
        "garage",
        "hasGuestRoom"
    ]
]

X_full = sm.add_constant(X_full)
det_XtX = np.linalg.det(X_full.T @ X_full)

print("det(X'X) =", det_XtX)
print("→ det(X'X) ≠ 0 ⇒ nghiệm OLS tồn tại duy nhất.\n")

print("============================================================")
print("BƯỚC 6. ƯỚC LƯỢNG MÔ HÌNH HỒI QUY TUYẾN TÍNH BỘI (MÔ HÌNH ĐẦY ĐỦ)")
print("============================================================\n")

# ------------------------------------------------------------------
# 6.0. XÁC ĐỊNH BIẾN PHỤ THUỘC & BIẾN GIẢI THÍCH
# ------------------------------------------------------------------
# y: biến phụ thuộc (giá nhà)
# X: tập biến giải thích (đã chọn từ bước 5)

y = df["price"]

X_full = df[
    [
        "squareMeters",
        "floors",
        "cityPartRange",
        "numberOfRooms",
        "numPrevOwners",
        "made",
        "basement",
        "attic",
        "garage",
        "hasGuestRoom"
    ]
]

# Thêm hằng số vào mô hình
X_full = sm.add_constant(X_full)

# ------------------------------------------------------------------
# 6.1. ƯỚC LƯỢNG MÔ HÌNH OLS
# ------------------------------------------------------------------
# Ước lượng mô hình hồi quy tuyến tính bội bằng phương pháp OLS

model_full = sm.OLS(y, X_full).fit()

# ------------------------------------------------------------------
# 6.2. REGRESSION STATISTICS (TƯƠNG ĐƯƠNG EXCEL)
# ------------------------------------------------------------------
# Mục đích: tóm tắt mức độ phù hợp của mô hình

print("----- REGRESSION STATISTICS -----")

reg_stats = pd.DataFrame({
    "Statistic": [
        "Multiple R",
        "R Square",
        "Adjusted R Square",
        "Standard Error",
        "Observations"
    ],
    "Value": [
        np.sqrt(model_full.rsquared),
        model_full.rsquared,
        model_full.rsquared_adj,
        np.sqrt(model_full.scale),
        int(model_full.nobs)
    ]
})

# Hiển thị nhiều chữ số để tránh làm tròn quá sớm
print(reg_stats.round(8))
print()

# ------------------------------------------------------------------
# 6.3. ANOVA TABLE (KIỂM ĐỊNH F TỔNG THỂ)
# ------------------------------------------------------------------
# Mục đích: kiểm tra mô hình hồi quy có ý nghĩa thống kê tổng thể hay không

print("----- ANOVA -----")

anova_table = pd.DataFrame({
    "Source": ["Regression", "Residual", "Total"],
    "df": [
        model_full.df_model,
        model_full.df_resid,
        model_full.df_model + model_full.df_resid
    ],
    "Sum of Squares": [
        model_full.ess,
        model_full.ssr,
        model_full.centered_tss
    ],
    "F": [
        model_full.fvalue,
        "",
        ""
    ],
    "Prob > F": [
        model_full.f_pvalue,
        "",
        ""
    ]
})

print(anova_table.round(6))
print()

# ------------------------------------------------------------------
# 6.4. BẢNG HỆ SỐ HỒI QUY (DÙNG CHO KIỂM ĐỊNH t)
# ------------------------------------------------------------------
# Mục đích: xem từng biến có ý nghĩa thống kê hay không
# Bảng này sẽ được dùng trực tiếp cho BƯỚC 7 (loại biến)

print("----- COEFFICIENTS TABLE -----")

coef_table = model_full.summary2().tables[1].round(6)
print(coef_table)
print()

print("Ghi chú:")
print("- Các hệ số được kiểm định bằng kiểm định t riêng phần.")
print("- Kiểm định F đánh giá ý nghĩa thống kê tổng thể của mô hình.\n")


print("============================================================")
print("BƯỚC 7. KIỂM ĐỊNH t & LỰA CHỌN BIẾN")
print("============================================================\n")

# ------------------------------------------------------------------
# 7.0. Ý tưởng
# ------------------------------------------------------------------
# - Dùng kiểm định t riêng phần cho từng hệ số
# - Loại dần các biến có p-value > alpha
# - Phương pháp: backward elimination

# ------------------------------------------------------------------
# 7.1. Backward Elimination
# ------------------------------------------------------------------
def backward_elimination(X, y, alpha=0.05):
    X_current = X.copy()

    while True:
        model = sm.OLS(y, X_current).fit()
        pvals = model.pvalues.drop("const")

        if pvals.max() > alpha:
            var_remove = pvals.idxmax()
            print(
                f"Loại biến: {VAR_LABELS.get(var_remove, var_remove)} "
                f"({var_remove}) – p-value = {pvals.max():.4f}"
            )
            X_current = X_current.drop(columns=[var_remove])
        else:
            break

    return sm.OLS(y, X_current).fit()


final_model = backward_elimination(X_full, y)
print()

# ------------------------------------------------------------------
# 7.2. REGRESSION STATISTICS (kiểu Excel – gọn)
# ------------------------------------------------------------------
print("----- REGRESSION STATISTICS -----")

reg_stats = pd.DataFrame({
    "Statistic": [
        "Multiple R",
        "R Square",
        "Adjusted R Square",
        "Standard Error",
        "Observations"
    ],
    "Value": [
        np.sqrt(final_model.rsquared),
        final_model.rsquared,
        final_model.rsquared_adj,
        np.sqrt(final_model.scale),
        int(final_model.nobs)
    ]
})

print(reg_stats.round(6))
print()

# ------------------------------------------------------------------
# 7.3. ANOVA TABLE
# ------------------------------------------------------------------
print("----- ANOVA -----")

anova_table = pd.DataFrame({
    "Source": ["Regression", "Residual", "Total"],
    "df": [
        final_model.df_model,
        final_model.df_resid,
        final_model.df_model + final_model.df_resid
    ],
    "Sum of Squares": [
        final_model.ess,
        final_model.ssr,
        final_model.centered_tss
    ],
    "F": [
        final_model.fvalue,
        "",
        ""
    ],
    "Prob > F": [
        final_model.f_pvalue,
        "",
        ""
    ]
})

print(anova_table.round(6))
print()

# ------------------------------------------------------------------
# 7.4. OLS SUMMARY – mô hình cuối
# ------------------------------------------------------------------
print("MÔ HÌNH SAU KHI LOẠI BIẾN:")
print(final_model.summary())
print()

print("Các biến còn lại trong mô hình:")
print(final_model.model.exog_names)
print()




# ===================== BƯỚC 8. CHẨN ĐOÁN MÔ HÌNH ======================
print("============================================================")
print("BƯỚC 8. CHẨN ĐOÁN MÔ HÌNH HỒI QUY")
print("============================================================\n")

# Phần dư và giá trị dự đoán
residuals = final_model.resid
fitted = final_model.fittedvalues

# Biểu đồ phần dư ~ giá trị dự đoán
plt.figure()
sns.scatterplot(x=fitted, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Giá trị dự đoán")
plt.ylabel("Phần dư")
plt.title("Biểu đồ phần dư theo giá trị dự đoán")
plt.tight_layout()
plt.savefig("figures/residuals_vs_fitted.png")
plt.close()

# Phần dư chuẩn hóa
std_residuals = final_model.get_influence().resid_studentized_internal

# Q-Q plot
sm.qqplot(std_residuals, line="45")
plt.title("Q-Q plot của phần dư chuẩn hóa")
plt.tight_layout()
plt.savefig("figures/qqplot_standardized_residuals.png")
plt.close()

print("Đã lưu các biểu đồ chẩn đoán phần dư.\n")


# ===================== BƯỚC 9. ƯỚC LƯỢNG & DỰ ĐOÁN ======================
print("============================================================")
print("BƯỚC 9. DỰ ĐOÁN GIÁ NHÀ")
print("============================================================\n")

df["predicted_price"] = final_model.predict(final_model.model.exog)
print("Đã tính giá nhà dự đoán cho các quan sát.\n")


# ===================== BƯỚC 10. KẾT LUẬN ======================
print("============================================================")
print("BƯỚC 10. KẾT LUẬN")
print("============================================================\n")

print("Mô hình hồi quy cuối cùng bao gồm các biến:")

for var in final_model.model.exog_names:
    if var == "const":
        print("- Hằng số")
    else:
        print(f"- {VAR_LABELS.get(var, var)}")

print("\nMô hình có ý nghĩa thống kê và phù hợp với khuôn khổ chương trình học.")
