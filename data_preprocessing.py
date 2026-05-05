# import numpy as np
# import pandas as pd
# import re
# from pathlib import Path
# from sklearn.preprocessing import StandardScaler
# import joblib

# # ==============================
# # 参数
# # ==============================

# DATA_DIR = Path("GraduationProject/data")
# OUTPUT_PATH = Path("GraduationProject/processed_data.csv")

# SMOOTH_WINDOW = 3   # ⚠️ 日数据不要太大（3~5）

# # ==============================
# # 列索引
# # ==============================

# COL_FLOW = 1
# COL_POWER = 2
# COL_CHEM = 5
# COL_PAC = 6
# COL_PAM = 7
# COL_MLSS = 8
# COL_COD = 15
# COL_LABEL = 35

# FEATURE_COLS = [
#     "flow","power","chemical","pac","pam","mlss","cod"
# ]

# # ==============================
# # 文件排序（年+月）
# # ==============================

# def get_year_month(file_name):
#     match = re.search(r'(\d{4}).*?(\d{1,2})', file_name)
#     if match:
#         return (int(match.group(1)), int(match.group(2)))
#     return (0, 0)


# def get_sorted_files():
#     files = list(DATA_DIR.glob("*.xls"))
#     return sorted(files, key=lambda x: get_year_month(x.name))


# # ==============================
# # 读取数据
# # ==============================

# # def load_data():

# #     print("========== 读取数据 ==========")

# #     files = get_sorted_files()
# #     data_list = []

# #     for file in files:

# #         print("读取:", file.name)

# #         df = pd.read_excel(file, engine="xlrd", header=None)
# #         df = df.iloc[3:-5]

# #         data = pd.DataFrame({
# #             "flow": df.iloc[:, COL_FLOW],
# #             "power": df.iloc[:, COL_POWER],
# #             "chemical": df.iloc[:, COL_CHEM],
# #             "pac": df.iloc[:, COL_PAC],
# #             "pam": df.iloc[:, COL_PAM],
# #             "mlss": df.iloc[:, COL_MLSS],
# #             "cod": df.iloc[:, COL_COD],
# #             "carbon": df.iloc[:, COL_LABEL]
# #         })

# #         data_list.append(data)

# #     data = pd.concat(data_list, ignore_index=True)

# #     return data

# def load_data():
#     print("========== 读取数据 ==========")

#     files = get_sorted_files()
#     data_list = []

#     for file in files:
#         print("读取:", file.name)

#         # ===== 1. 解析 年月 =====
#         year, month = get_year_month(file.name)

#         df = pd.read_excel(file, header=None)

#         # ===== 2. 去掉前几行（表头）=====
#         df = df.iloc[3:].reset_index(drop=True)

#         # ===== 3. 自动过滤“非数据行” =====
#         # 判断第一列是不是数字
#         df = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notna()]

#         # ===== 4. 重建“日期” =====
#         df = df.reset_index(drop=True)
#         df["day"] = df.index + 1
#         df["date"] = pd.to_datetime({
#             "year": year,
#             "month": month,
#             "day": df["day"]
#         }, errors="coerce")

#         # ===== 5. 构建特征 =====
#         data = pd.DataFrame({
#             "date": df["date"],
#             "flow": df.iloc[:, COL_FLOW],
#             "power": df.iloc[:, COL_POWER],
#             "chemical": df.iloc[:, COL_CHEM],
#             "pac": df.iloc[:, COL_PAC],
#             "pam": df.iloc[:, COL_PAM],
#             "mlss": df.iloc[:, COL_MLSS],
#             "cod": df.iloc[:, COL_COD],
#             "carbon": df.iloc[:, COL_LABEL]
#         })
#         data_list.append(data)

#     # ===== 6. 拼接 =====
#     data = pd.concat(data_list, ignore_index=True)

#     # ===== 7. 按时间排序（关键）=====
#     data = data.sort_values("date").reset_index(drop=True)
#     data['month'] = data['date'].dt.month
#     data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
#     data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
#     return data
# # ==============================
# # 数据清洗
# # ==============================

# def clean_data(data):

#     print("========== 数据清洗 ==========")

#     for col in FEATURE_COLS + ["carbon"]:
#         # 把一列强制转换成“数值类型”，无法转换的变成 NaN
#         data[col] = pd.to_numeric(data[col], errors="coerce")

#     # data = data.dropna()  # 删除有NaN的行
#     data = data.interpolate()

#     data = data.reset_index(drop=True)

#     return data


# # ==============================
# # 异常值处理（IQR，一次性过滤，避免数据崩掉）
# # ==============================

# def remove_outliers(data):

#     print("========== 异常值处理 ==========")

#     mask = pd.Series([True] * len(data))

#     for col in FEATURE_COLS + ["carbon"]:

#         Q1 = data[col].quantile(0.25)
#         Q3 = data[col].quantile(0.75)
#         IQR = Q3 - Q1

#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR

#         mask &= (data[col] >= lower) & (data[col] <= upper)

#     data = data[mask].reset_index(drop=True)

#     return data


# # ==============================
# # 平滑（轻度）
# # ==============================

# def smooth_data(data):

#     print("========== 数据平滑 ==========")

#     for col in FEATURE_COLS + ["carbon"]:
#         data[col] = data[col].rolling(
#             window=SMOOTH_WINDOW,
#             min_periods=1,
#             center=False
#         ).mean()

#     return data


# # ==============================
# # 保存
# # ==============================

# def save_data(data):

#     print("========== 保存数据 ==========")

#     data.to_csv(OUTPUT_PATH, index=False)

#     print("保存路径:", OUTPUT_PATH)


# # ==============================
# # 主流程
# # ==============================

# def main():

#     data = load_data()

#     data = clean_data(data)

#     data = remove_outliers(data)

#     data = smooth_data(data)

#     save_data(data)

#     print("========== 预处理完成 ==========")


# if __name__ == "__main__":
#     main()


############################################################

import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================
# 参数（路径相对本文件，任意 cwd 可运行）
# ==============================
_PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = _PROJECT_ROOT / "data"
OUTPUT_PATH = _PROJECT_ROOT / "processed" / "processed_data.csv"

# 论文 §2.2：滚动均值窗口长度 5（特征与目标一致）
SMOOTH_WINDOW = 5

# ==============================
# 列索引（基于月报表结构）
# ==============================
# 基础运行数据
COL_FLOW = 1          # 处理水量
COL_POWER = 2         # 电量
COL_SLUDGE_WET = 3    # 湿泥量
COL_SLUDGE_MOISTURE = 4  # 脱水污泥含水率
COL_CHEM = 5          # 药剂量(kg) -> chemical
COL_PAC = 6
COL_PAM = 7
COL_MLSS = 8
COL_POWER_RATE = 10   # 电耗率
COL_CHEM_RATE = 11    # 脱水药耗率

# 水质指标（进出水）
COL_SS_IN = 12
COL_SS_OUT = 13
COL_BOD_IN = 14
COL_BOD_OUT = 15
COL_COD_IN = 16       # 原为cod
COL_COD_OUT = 17
COL_COD_REDUCTION = 19
COL_PH_IN = 20
COL_PH_OUT = 21
COL_TN_IN = 22
COL_TN_OUT = 23
COL_TP_IN = 24
COL_TP_OUT = 25
COL_NH3_IN = 26
COL_NH3_OUT = 27
COL_NH3_REDUCTION = 29
COL_COLOR_IN = 30
COL_COLOR_OUT = 31

# 目标变量：总间接碳排 kgCO2/m3（列索引需确认，在样例中约33）
COL_CARBON = 35       # 根据你的原脚本，目标列索引为35

# 所有特征列（不包括目标，但包括新增的）
FEATURE_COLS = [
    "flow", "power", "sludge_wet", "sludge_moisture",
    "chemical", "pac", "pam", "mlss",
    "power_rate", "chemical_rate",
    "ss_in", "ss_out", "bod_in", "bod_out",
    "cod_in", "cod_out", "cod_reduction",
    "ph_in", "ph_out", "tn_in", "tn_out",
    "tp_in", "tp_out", "nh3_in", "nh3_out", "nh3_reduction",
    "color_in", "color_out",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos",
]

# ==============================
# 文件排序（年+月）
# ==============================
def get_year_month(file_name):
    match = re.search(r'(\d{4}).*?(\d{1,2})', file_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)

def get_sorted_files():
    files = list(DATA_DIR.glob("*.xls"))
    return sorted(files, key=lambda x: get_year_month(x.name))

# ==============================
# 读取数据（增强版）
# ==============================
def load_data():
    print("========== 读取数据 ==========")
    files = get_sorted_files()
    data_list = []

    for file in files:
        print("读取:", file.name)
        year, month = get_year_month(file.name)

        df = pd.read_excel(file, header=None)
        # 去掉前几行表头（原脚本跳过了3行）
        df = df.iloc[3:].reset_index(drop=True)

        # 自动过滤非数据行（第一列非数字的行）
        df = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').notna()]

        df = df.reset_index(drop=True)
        df["day"] = df.index + 1
        df["date"] = pd.to_datetime({
            "year": year,
            "month": month,
            "day": df["day"]
        }, errors="coerce")

        # 构建DataFrame，处理可能缺失的列（有些文件可能列数不足，用NaN填充）
        data = pd.DataFrame({
            "date": df["date"],
            "flow": df.iloc[:, COL_FLOW],
            "power": df.iloc[:, COL_POWER],
            "sludge_wet": df.iloc[:, COL_SLUDGE_WET] if df.shape[1] > COL_SLUDGE_WET else np.nan,
            "sludge_moisture": df.iloc[:, COL_SLUDGE_MOISTURE] if df.shape[1] > COL_SLUDGE_MOISTURE else np.nan,
            "chemical": df.iloc[:, COL_CHEM],
            "pac": df.iloc[:, COL_PAC],
            "pam": df.iloc[:, COL_PAM],
            "mlss": df.iloc[:, COL_MLSS],
            "power_rate": df.iloc[:, COL_POWER_RATE] if df.shape[1] > COL_POWER_RATE else np.nan,
            "chemical_rate": df.iloc[:, COL_CHEM_RATE] if df.shape[1] > COL_CHEM_RATE else np.nan,
            "ss_in": df.iloc[:, COL_SS_IN] if df.shape[1] > COL_SS_IN else np.nan,
            "ss_out": df.iloc[:, COL_SS_OUT] if df.shape[1] > COL_SS_OUT else np.nan,
            "bod_in": df.iloc[:, COL_BOD_IN] if df.shape[1] > COL_BOD_IN else np.nan,
            "bod_out": df.iloc[:, COL_BOD_OUT] if df.shape[1] > COL_BOD_OUT else np.nan,
            "cod_in": df.iloc[:, COL_COD_IN],
            "cod_out": df.iloc[:, COL_COD_OUT] if df.shape[1] > COL_COD_OUT else np.nan,
            "cod_reduction": df.iloc[:, COL_COD_REDUCTION] if df.shape[1] > COL_COD_REDUCTION else np.nan,
            "ph_in": df.iloc[:, COL_PH_IN] if df.shape[1] > COL_PH_IN else np.nan,
            "ph_out": df.iloc[:, COL_PH_OUT] if df.shape[1] > COL_PH_OUT else np.nan,
            "tn_in": df.iloc[:, COL_TN_IN] if df.shape[1] > COL_TN_IN else np.nan,
            "tn_out": df.iloc[:, COL_TN_OUT] if df.shape[1] > COL_TN_OUT else np.nan,
            "tp_in": df.iloc[:, COL_TP_IN] if df.shape[1] > COL_TP_IN else np.nan,
            "tp_out": df.iloc[:, COL_TP_OUT] if df.shape[1] > COL_TP_OUT else np.nan,
            "nh3_in": df.iloc[:, COL_NH3_IN] if df.shape[1] > COL_NH3_IN else np.nan,
            "nh3_out": df.iloc[:, COL_NH3_OUT] if df.shape[1] > COL_NH3_OUT else np.nan,
            "nh3_reduction": df.iloc[:, COL_NH3_REDUCTION] if df.shape[1] > COL_NH3_REDUCTION else np.nan,
            "color_in": df.iloc[:, COL_COLOR_IN] if df.shape[1] > COL_COLOR_IN else np.nan,
            "color_out": df.iloc[:, COL_COLOR_OUT] if df.shape[1] > COL_COLOR_OUT else np.nan,
            "carbon": df.iloc[:, COL_CARBON]
        })
        data_list.append(data)

    data = pd.concat(data_list, ignore_index=True)
    data = data.sort_values("date").reset_index(drop=True)

    # 月份 / 年内日序（强化季节性，间接碳排多与季节运行模式相关）
    data['month'] = data['date'].dt.month
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    doy = data['date'].dt.dayofyear.astype(float)
    data['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    data['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    dow = data['date'].dt.dayofweek.astype(float)
    data['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    data['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    return data

# ==============================
# 数据清洗（转为数值、插值）
# ==============================
def clean_data(data):
    print("========== 数据清洗 ==========")
    for col in FEATURE_COLS + ["carbon"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    # 使用线性插值填充缺失值
    data = data.interpolate(method='linear', limit_direction='both')
    data = data.reset_index(drop=True)
    return data

# ==============================
# 平滑（论文 §2.2，Fig.1 流程）
# ==============================
def smooth_data(data):
    print("========== 滚动均值平滑（窗口=5）==========")
    for col in FEATURE_COLS + ["carbon"]:
        data[col] = data[col].rolling(
            window=SMOOTH_WINDOW, min_periods=1, center=False
        ).mean()
    return data

# ==============================
# 保存
# ==============================
def save_data(data):
    print("========== 保存数据 ==========")
    data.to_csv(OUTPUT_PATH, index=False)
    print("保存路径:", OUTPUT_PATH)

# ==============================
# 主流程
# ==============================
def main():
    data = load_data()
    data = clean_data(data)
    data = smooth_data(data)
    save_data(data)
    print("========== 预处理完成（插值 + 滚动均值5，见论文 §2.2）==========")

if __name__ == "__main__":
    main()