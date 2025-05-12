import os
import math
import streamlit as st
import pandas as pd
import io
import zipfile
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles # 벤다이어그램 사용시 주석 해제
# import matplotlib # 중복될 수 있음

# Selenium 관련 import
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.chrome.service import Service as ChromeService # webdriver_manager 사용 시
from webdriver_manager.chrome import ChromeDriverManager # webdriver_manager 사용 시
import re
# import os # ZIP 파일 생성 시 BytesIO를 사용하므로 파일 시스템 직접 접근은 최소화

st.set_page_config(page_title="CSV & Webtoon Utility App", layout="wide")
st.title("CSV 조작 및 웹툰 정보 추출 앱")

# ------------------------------------------------------------------------------
# A. CSV 업로드 처리
# ------------------------------------------------------------------------------
st.subheader("1) CSV 업로드")

uploaded_files = st.file_uploader(
    "여기에 CSV 파일들을 드래그하거나, 'Browse files' 버튼을 눌러 선택하세요.",
    type=["csv"],
    accept_multiple_files=True
)

if "csv_dataframes" not in st.session_state:
    st.session_state["csv_dataframes"] = {}
if "file_names" not in st.session_state:
    st.session_state["file_names"] = {}

if st.button("CSV 로드하기"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                st.session_state["csv_dataframes"][uploaded_file.name] = df
                if uploaded_file.name not in st.session_state["file_names"]:
                    st.session_state["file_names"][uploaded_file.name] = uploaded_file.name
                st.success(f"{uploaded_file.name}: 업로드 & 로드 성공 (행 수: {len(df)})")
            except Exception as e:
                st.error(f"{uploaded_file.name}: 로드 실패 - {e}")
    else:
        st.warning("아직 업로드된 파일이 없습니다.")

# ------------------------------------------------------------------------------
# B. 업로드된 파일 목록 표시 (파일명 변경 가능)
# ------------------------------------------------------------------------------
if st.session_state["csv_dataframes"]:
    st.write("### 업로드된 파일 목록")
    keys_to_delete = [] # 삭제할 키를 임시 저장할 리스트
    for original_name in list(st.session_state["csv_dataframes"].keys()): # 순회 중 변경 방지를 위해 list로 복사
        if original_name not in st.session_state["csv_dataframes"]: # 이미 삭제된 경우 건너뛰기
            continue
        df = st.session_state["csv_dataframes"][original_name]
        col1, col2, col3, col_del_btn = st.columns([2, 1.5, 1, 0.5])


        with col1:
            new_name = st.text_input(
                f"파일명 변경 ({original_name})",
                value=st.session_state["file_names"].get(original_name, original_name),
                key=f"file_name_{original_name}"
            )
            other_file_names = [v for k, v in st.session_state["file_names"].items() if k != original_name]
            if new_name in other_file_names:
                st.warning(f"'{new_name}' 파일명이 이미 존재합니다. 다른 이름을 입력해주세요.", icon="⚠️")
            else:
                st.session_state["file_names"][original_name] = new_name

        with col2:
            st.write(f"행 수: {len(df)}")

        with col3:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, header=False)
            st.download_button(
                label="다운로드",
                data=csv_buffer.getvalue(),
                file_name=st.session_state["file_names"].get(original_name, original_name),
                mime="text/csv",
                key=f"download_btn_{original_name}"
            )
        with col_del_btn:
            if st.button("삭제", key=f"delete_btn_{original_name}"):
                keys_to_delete.append(original_name)
                st.rerun() # UI 즉시 업데이트
    
    # 삭제 로직 실행
    if keys_to_delete:
        for key_del in keys_to_delete:
            if key_del in st.session_state["csv_dataframes"]:
                del st.session_state["csv_dataframes"][key_del]
            if key_del in st.session_state["file_names"]:
                del st.session_state["file_names"][key_del]
        st.success(f"{len(keys_to_delete)}개 파일 삭제 완료.")
        # st.rerun() # 삭제 후 UI 갱신 (위의 rerun으로 이미 처리될 수 있음)


# ------------------------------------------------------------------------------
# C. 공통 함수
# ------------------------------------------------------------------------------
def get_uid_set(csv_key):
    df = st.session_state["csv_dataframes"][csv_key]
    return set(df.iloc[:, 0].astype(str))

def save_to_session_and_download(result_df, file_name="result.csv"):
    unique_file_name = file_name
    counter = 1
    current_filenames_in_use = list(st.session_state["file_names"].values())
    
    while unique_file_name in current_filenames_in_use:
        base, ext = os.path.splitext(unique_file_name) # os.path.splitext 사용
        unique_file_name = f"{base}_{counter}{ext}"
        counter += 1

    original_key_for_new_file = unique_file_name 
    st.session_state["csv_dataframes"][original_key_for_new_file] = result_df
    st.session_state["file_names"][original_key_for_new_file] = unique_file_name

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False, header=False) # CSV는 헤더 없이 저장
    st.download_button(
        label=f"{unique_file_name} 다운로드",
        data=csv_buffer.getvalue(),
        file_name=unique_file_name,
        mime="text/csv",
        key=f"download_generated_{unique_file_name}" # 고유 키 생성
    )
    st.success(f"결과 CSV '{unique_file_name}' 세션에 추가 완료!")

# ------------------------------------------------------------------------------
# D. 교집합
# ------------------------------------------------------------------------------
st.subheader("2) 교집합 (Intersection)")
selected_for_intersect = st.multiselect(
    "교집합 대상 CSV 선택 (2개 이상)",
    list(st.session_state["file_names"].values()),
    key="intersect_select"
)
if st.button("교집합 실행하기", key="intersect_run"):
    if len(selected_for_intersect) < 2:
        st.error("교집합은 2개 이상 선택해야 합니다.")
    else:
        base = None
        for current_file_name in selected_for_intersect:
            original_key = [k for k, v in st.session_state["file_names"].items() if v == current_file_name][0]
            if base is None:
                base = get_uid_set(original_key)
            else:
                base = base.intersection(get_uid_set(original_key))
        st.write(f"교집합 결과 UID 수: {len(base)}")
        result_df = pd.DataFrame(sorted(list(base)))
        save_to_session_and_download(result_df, "result_intersection.csv")

# ------------------------------------------------------------------------------
# E. 조합 (합집합 + 제외)
# ------------------------------------------------------------------------------
st.subheader("3) 조합하기 (포함/제외)")
col1_comb, col2_comb = st.columns(2)
with col1_comb:
    plus_list = st.multiselect("포함(+)할 CSV들", list(st.session_state["file_names"].values()), key="comb_plus")
with col2_comb:
    minus_list = st.multiselect("제외(-)할 CSV들", list(st.session_state["file_names"].values()), key="comb_minus")
if st.button("조합하기 실행", key="comb_run"):
    if not plus_list:
        st.error("최소 1개 이상 '포함' 리스트를 지정해야 합니다.")
    else:
        plus_set = set()
        for p in plus_list:
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == p][0]
            plus_set = plus_set.union(get_uid_set(orig_key))
        minus_set = set()
        for m in minus_list:
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == p][0] # 버그 수정: p -> m
            minus_set = minus_set.union(get_uid_set(orig_key))
        final = plus_set - minus_set
        st.write(f"조합 결과 UID 수: {len(final)}")
        result_df = pd.DataFrame(sorted(list(final)))
        save_to_session_and_download(result_df, "result_combination.csv")


# ------------------------------------------------------------------------------
# F. N번 이상 또는 정확히 N번 CSV에서 등장하는 UID 추출
# ------------------------------------------------------------------------------
st.subheader("4) N번 이상 / 정확히 N번 등장하는 UID 추출")
selected_for_nplus = st.multiselect(
    "대상 CSV 선택 (1개 이상)",
    list(st.session_state["file_names"].values()),
    key="nplus_select"
)
threshold = st.number_input(
    "몇 개의 CSV 파일에서 등장하는 UID를 찾을까요?",
    min_value=1, value=2, step=1, key="nplus_threshold"
)
condition_option = st.radio(
    "추출 조건 선택",
    ("N번 이상 등장", "정확히 N번 등장"),
    key="nplus_condition"
)
if st.button("UID 추출 실행", key="nplus_run"):
    if not selected_for_nplus:
        st.error("최소 1개 이상의 CSV 파일을 선택해주세요.")
    else:
        if threshold > len(selected_for_nplus) and len(selected_for_nplus) > 0:
            st.warning(f"선택한 CSV 파일은 {len(selected_for_nplus)}개인데, {threshold}개 이상을 선택하셨습니다. 결과가 없을 수 있습니다.")
        uid_count = {}
        for fname in selected_for_nplus:
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == fname][0]
            current_uids = get_uid_set(orig_key)
            for uid in current_uids:
                uid_count[uid] = uid_count.get(uid, 0) + 1
        if condition_option == "N번 이상 등장":
            valid_uids = [uid for uid, cnt in uid_count.items() if cnt >= threshold]
            st.write(f"{threshold}개 이상 등장하는 UID 수: {len(valid_uids)}")
        else:
            valid_uids = [uid for uid, cnt in uid_count.items() if cnt == threshold]
            st.write(f"정확히 {threshold}개 등장하는 UID 수: {len(valid_uids)}")
        if valid_uids:
            result_df = pd.DataFrame(sorted(valid_uids))
            filename = "result_n_or_more.csv" if condition_option == "N번 이상 등장" else "result_exactly_n.csv" # 파일명 수정
            save_to_session_and_download(result_df, filename)
        else:
            st.warning("해당 조건을 만족하는 UID가 없습니다.")

# ------------------------------------------------------------------------------
# G. 중복 제거 (Unique)
# ------------------------------------------------------------------------------
st.subheader("5) 중복 제거")
unique_target = st.selectbox(
    "중복 제거할 CSV 선택",
    ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values()),
    key="unique_select"
)
if st.button("중복 제거 실행하기", key="unique_run"):
    if unique_target == "--- 선택하세요 ---":
        st.error("중복 제거할 파일을 선택해주세요.")
    else:
        orig_key = [k for k, v in st.session_state["file_names"].items() if v == unique_target][0]
        df = st.session_state["csv_dataframes"][orig_key]
        df_unique = df.drop_duplicates(keep="first")
        st.write(f"원본 행 수: {len(df)}, 중복 제거 후 행 수: {len(df_unique)}")
        base_name, ext = os.path.splitext(unique_target) # os.path.splitext 사용
        save_to_session_and_download(df_unique, f"{base_name}_unique{ext}") # 파일명 수정


# ------------------------------------------------------------------------------
# H. 랜덤 추출
# ------------------------------------------------------------------------------
st.subheader("6) 랜덤 추출")
random_targets = st.multiselect(
    "랜덤 추출 대상 CSV 선택 (1개 이상)",
    list(st.session_state["file_names"].values()),
    key="random_select"
)
sample_size = st.number_input("랜덤 추출 개수", min_value=1, value=10, step=1, key="random_sample_size")
if st.button("랜덤 추출 실행하기", key="random_run"):
    if not random_targets:
        st.error("최소 1개 이상의 파일을 선택해주세요.")
    else:
        combined_df = pd.DataFrame()
        for rt in random_targets:
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == rt][0]
            combined_df = pd.concat([combined_df, st.session_state["csv_dataframes"][orig_key]], ignore_index=True)
        if len(combined_df) == 0:
            st.warning("선택한 파일들의 데이터가 비어있어 랜덤 추출을 할 수 없습니다.")
        elif sample_size > len(combined_df):
            st.warning(f"랜덤 추출 개수({sample_size})가 총 행 수({len(combined_df)})보다 많습니다. 가능한 최대 개수만큼 추출합니다.")
            sample_size = len(combined_df)
        if len(combined_df) > 0: # 데이터가 있을 때만 sample 실행
            random_sample = combined_df.sample(n=sample_size, random_state=None) # random_state=None으로 매번 다르게
            st.write(f"통합된 행 수: {len(combined_df)}, 랜덤 추출 개수: {len(random_sample)}")
            save_to_session_and_download(random_sample, "result_random_sample.csv") # 파일명 수정

# ------------------------------------------------------------------------------
# I. Bingo 당첨자 추출 (기존 코드와 유사하게 유지, Streamlit 요소에 key 추가)
# ------------------------------------------------------------------------------
st.subheader("7) 빙고 당첨자 추출") # 번호 수정
bingo_size_options = ["2x2", "3x3", "4x4", "5x5"]
default_bingo_size_idx = 1
if "bingo_size_selection" in st.session_state:
    try:
        default_bingo_size_idx = bingo_size_options.index(st.session_state.bingo_size_selection)
    except ValueError:
        st.session_state.bingo_size_selection = bingo_size_options[default_bingo_size_idx]

bingo_size_selection = st.selectbox(
    "빙고판 크기 선택", bingo_size_options, index=default_bingo_size_idx, key="bingo_size_selector_key_main"
)
if st.session_state.get("bingo_size_selection") != bingo_size_selection:
    st.session_state.bingo_size_selection = bingo_size_selection

size_map = {"2x2": 2, "3x3": 3, "4x4": 4, "5x5": 5}
n = size_map[bingo_size_selection]

cell_files = [None] * (n * n)
st.markdown("### 빙고판 구성 (번호 순서대로 CSV 선택)")
available_files_for_bingo = ["---"] + list(st.session_state.get("file_names", {}).values())

if len(available_files_for_bingo) > 1:
    for r_bingo in range(n):
        cols_bingo = st.columns(n)
        for c_bingo in range(n):
            idx_bingo = r_bingo * n + c_bingo
            with cols_bingo[c_bingo]:
                cell_session_key = f"bingo_cell_selection_{n}_{idx_bingo}"
                selectbox_key = f"selectbox_cell_ui_{n}_{idx_bingo}"
                current_selection_for_cell = st.session_state.get(cell_session_key, "---")
                current_selection_idx = 0
                if current_selection_for_cell in available_files_for_bingo:
                    current_selection_idx = available_files_for_bingo.index(current_selection_for_cell)
                else:
                    st.session_state[cell_session_key] = "---"
                option = st.selectbox(
                    f"{idx_bingo+1}번 칸", available_files_for_bingo, index=current_selection_idx, key=selectbox_key
                )
                if st.session_state.get(cell_session_key) != option:
                    st.session_state[cell_session_key] = option
                if option != "---":
                    cell_files[idx_bingo] = option
else:
    st.warning("빙고판을 구성하려면 먼저 CSV 파일을 업로드하고 로드해주세요.")

def get_bingo_lines(n_val):
    lines = []
    for r_val in range(n_val): lines.append([r_val * n_val + c_val for c_val in range(n_val)])
    for c_val in range(n_val): lines.append([r_val * n_val + c_val for r_val in range(n_val)])
    lines.append([i * n_val + i for i in range(n_val)])
    lines.append([i * n_val + (n_val - 1 - i) for i in range(n_val)])
    return lines

if st.button("당첨자 추출하기", key="run_bingo_extraction_key_main"):
    current_cell_files = [None] * (n * n)
    for r_bingo_final in range(n):
        for c_bingo_final in range(n):
            idx_bingo_final = r_bingo_final * n + c_bingo_final
            cell_session_key_final = f"bingo_cell_selection_{n}_{idx_bingo_final}"
            selected_file_for_cell = st.session_state.get(cell_session_key_final, "---")
            if selected_file_for_cell != "---":
                current_cell_files[idx_bingo_final] = selected_file_for_cell
            else:
                current_cell_files[idx_bingo_final] = None
    
    if not any(f for f in current_cell_files if f is not None):
        st.error("빙고판에 CSV 파일을 하나 이상 선택해주세요.")
    else:
        uid_sets, counts = [], []
        file_names_map = st.session_state.get("file_names", {})
        csv_dataframes = st.session_state.get("csv_dataframes", {})
        for display_name_in_cell in current_cell_files:
            if display_name_in_cell:
                original_file_key = next((k for k, v in file_names_map.items() if v == display_name_in_cell), None)
                if original_file_key and original_file_key in csv_dataframes:
                    df_bingo = csv_dataframes[original_file_key]
                    if not df_bingo.empty and df_bingo.shape[1] > 0:
                        s = set(df_bingo.iloc[:, 0].astype(str))
                        uid_sets.append(s)
                        counts.append(len(s))
                    else:
                        uid_sets.append(set()); counts.append(0)
                else:
                    uid_sets.append(set()); counts.append(0)
                    st.warning(f"경고: 빙고 셀 파일 '{display_name_in_cell}'에 대한 데이터를 찾을 수 없습니다.")
            else:
                uid_sets.append(set()); counts.append(0)
        lines = get_bingo_lines(n)
        line_sets = []
        for ln_indices in lines:
            inter = set()
            if ln_indices and uid_sets[ln_indices[0]]:
                inter = uid_sets[ln_indices[0]].copy()
                for i_idx in ln_indices[1:]:
                    if uid_sets[i_idx]: inter &= uid_sets[i_idx]
                    else: inter = set(); break
            line_sets.append(inter)
        user_count = defaultdict(int)
        for s_set in line_sets:
            for uid_val in s_set: user_count[uid_val] += 1
        
        fig, ax = plt.subplots(figsize=(n * 2, n * 2))
        ax.set_xlim(0, n); ax.set_ylim(0, n)
        ax.set_xticks(range(n + 1)); ax.set_yticks(range(n + 1))
        ax.grid(True, zorder=0)
        cmap = plt.get_cmap('tab20', len(lines))
        colors = [cmap(i) for i in range(len(lines))]
        for i, ln_indices in enumerate(lines):
            xs = [(idx % n) + 0.5 for idx in ln_indices]
            ys = [n - (idx // n) - 0.5 for idx in ln_indices]
            ax.plot(xs, ys, color=colors[i], linewidth=3, zorder=1, alpha=0.7)
        for idx_patch in range(n * n):
            r_patch, c_patch = divmod(idx_patch, n)
            ax.add_patch(Rectangle((c_patch, n - r_patch - 1), 1, 1, fill=False, edgecolor='gray', linewidth=1, zorder=2))
            file_display_name = current_cell_files[idx_patch] or '---'
            if len(file_display_name) > 12 and n <= 3: file_display_name = file_display_name[:10] + "\n" + file_display_name[10:20]
            elif len(file_display_name) > 18 : file_display_name = file_display_name[:15] + "\n" + file_display_name[15:30]
            txt = f"{file_display_name}\n({counts[idx_patch]})"
            ax.text(c_patch + 0.5, n - r_patch - 0.5, txt, ha='center', va='center', fontsize=6 if n>=4 else 8,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.9), zorder=3)
        base_offset = 0.55 if n <=3 else 0.65
        for i, ln_indices in enumerate(lines):
            if line_sets[i]:
                xs = [(idx % n) + 0.5 for idx in ln_indices]; ys = [n - (idx // n) - 0.5 for idx in ln_indices]
                if not xs or not ys: continue
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                angle_rad = math.atan2(my - n / 2, mx - n / 2) if n > 0 else 0
                current_offset = base_offset
                is_diag1 = (ln_indices == [k_val * n + k_val for k_val in range(n)])
                is_diag2 = (ln_indices == [k_val * n + (n - 1 - k_val) for k_val in range(n)])
                if (is_diag1 or is_diag2) and n > 2: current_offset = base_offset * 1.3
                dx = current_offset * math.cos(angle_rad); dy = current_offset * math.sin(angle_rad)
                label_x, label_y = mx + dx, my + dy
                ax.text(label_x, label_y, str(len(line_sets[i])), color=colors[i], fontsize=8 if n >=4 else 10,
                        fontweight='bold', ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor=colors[i], alpha=1.0, pad=0.3), zorder=5)
        ax.set_title(f"{n}x{n} 빙고판 시각화", fontsize=12); ax.axis('off')
        st.pyplot(fig)
        line_names = []
        for r_idx in range(n): line_names.append(f"가로 {r_idx+1}")
        for c_idx in range(n): line_names.append(f"세로 {c_idx+1}")
        line_names.append("대각선 \\"); line_names.append("대각선 /")
        info_data = []
        for idx, ln_indices_info in enumerate(lines):
            current_line_name = line_names[idx] if idx < len(line_names) else f"라인 {idx+1}"
            info_data.append({
                "빙고 라인": current_line_name, "셀 번호 (0-indexed)": str(ln_indices_info), "달성자 수": len(line_sets[idx])
            })
        st.dataframe(pd.DataFrame(info_data))
        st.markdown("---")
        zip_lines_buf = io.BytesIO(); lines_zip_created = False
        with zipfile.ZipFile(zip_lines_buf, mode='w') as zf_lines:
            for i, line_name_zip in enumerate(line_names):
                uids_in_line = list(line_sets[i])
                if uids_in_line:
                    dfw = pd.DataFrame(uids_in_line, columns=['UID']).sort_values(by='UID').reset_index(drop=True)
                    safe_line_name = line_name_zip.replace(" ", "_").replace("\\", "diag1").replace("/", "diag2")
                    zf_lines.writestr(f"bingo_line_{safe_line_name}_winners.csv", dfw.to_csv(index=False, header=True).encode('utf-8-sig'))
                    lines_zip_created = True
        if lines_zip_created:
            zip_lines_buf.seek(0)
            st.download_button(
                label="각 라인별 당첨자 목록 ZIP 다운로드", data=zip_lines_buf,
                file_name=f"bingo_{n}x{n}_lines_winners.zip", mime="application/zip", key="download_bingo_lines_zip_key_main"
            )
        else: st.info("빙고를 달성한 라인이 없어 '각 라인별 당첨자 목록'을 다운로드할 파일이 없습니다.")
        st.markdown("---")
        st.markdown("### N-빙고 달성자 목록 (정확히 N개 라인 달성)")
        if user_count:
            winners_by_exact_bingo_count = defaultdict(list)
            for uid, count_val in user_count.items():
                if count_val > 0: winners_by_exact_bingo_count[count_val].append(uid)
            if winners_by_exact_bingo_count:
                zip_exact_buf = io.BytesIO(); exact_zip_created = False
                with zipfile.ZipFile(zip_exact_buf, mode='w') as zf_exact:
                    max_possible_bingos = len(lines)
                    for num_bingos in range(1, max_possible_bingos + 1):
                        if num_bingos in winners_by_exact_bingo_count:
                            uids_for_exact_count = sorted(list(winners_by_exact_bingo_count[num_bingos]))
                            if uids_for_exact_count:
                                df_exact_winners = pd.DataFrame(uids_for_exact_count, columns=['UID'])
                                csv_content = df_exact_winners.to_csv(index=False, header=True).encode('utf-8-sig')
                                zf_exact.writestr(f"bingo_winners_{num_bingos}_lines.csv", csv_content); exact_zip_created = True
                if exact_zip_created:
                    zip_exact_buf.seek(0)
                    st.download_button(
                        label=f"N-빙고 달성자 목록 ZIP 다운로드 (빙고 수별 파일)", data=zip_exact_buf,
                        file_name=f"bingo_{n}x{n}_winners_by_exact_line_count.zip", mime="application/zip", key="download_exact_bingo_count_zip_key_main"
                    )
                else: st.info("N-빙고 달성자 데이터는 있으나, 다운로드할 파일을 생성하지 못했습니다.")
            else: st.info("빙고를 1개 이상 달성한 사용자가 없어 N-빙고별 목록을 생성할 수 없습니다.")
        else: st.info("빙고를 달성한 사용자가 없어 N-빙고별 목록을 생성할 수 없습니다.")
        st.markdown("---")
        st.markdown("### N개 이상 빙고 달성자 목록")
        num_input_key_at_least_n = f"min_bingo_lines_input_{n}"
        if num_input_key_at_least_n not in st.session_state: st.session_state[num_input_key_at_least_n] = 1
        max_bingo_val = len(lines) if lines else 1
        min_bingo_lines = st.number_input(
            "추출할 최소 빙고 라인 수 (N)", min_value=1, max_value=max_bingo_val,
            value=st.session_state[num_input_key_at_least_n], step=1, key=num_input_key_at_least_n,
            on_change=lambda: st.session_state.update({num_input_key_at_least_n: st.session_state[num_input_key_at_least_n]})
        )
        if st.button("N개 이상 빙고 달성자 목록 보기", key="view_multi_bingo_winners_key_main"):
            if user_count:
                current_min_lines = st.session_state[num_input_key_at_least_n]
                winners_n_bingo = {uid: count_val for uid, count_val in user_count.items() if count_val >= current_min_lines}
                if winners_n_bingo:
                    df_multi_bingo = pd.DataFrame(list(winners_n_bingo.items()), columns=['UID', '달성 라인 수'])
                    df_multi_bingo = df_multi_bingo.sort_values(by=['달성 라인 수', 'UID'], ascending=[False, True]).reset_index(drop=True)
                    st.write(f"#### {current_min_lines}개 이상 빙고 달성자 ({len(df_multi_bingo)}명)")
                    st.dataframe(df_multi_bingo)
                    csv_multi_bingo = df_multi_bingo.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label=f"{current_min_lines}개 이상 빙고 달성자 CSV 다운로드", data=csv_multi_bingo,
                        file_name=f"bingo_winners_{current_min_lines}_or_more_lines.csv", mime="text/csv", key="download_multi_bingo_winners_csv_key_main"
                    )
                else: st.info(f"{current_min_lines}개 이상의 빙고를 달성한 사용자가 없습니다.")
            else: st.info("빙고를 달성한 사용자가 없어 N개 이상 빙고 달성자 목록을 생성할 수 없습니다.")

# ------------------------------------------------------------------------------
# J. 특정 열(컬럼) 삭제하기
# ------------------------------------------------------------------------------
st.subheader("8) 특정 열(컬럼) 삭제하기") # 번호 수정
column_delete_target = st.selectbox(
    "열 삭제할 CSV를 선택하세요",
    ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values()),
    key="col_delete_select_main"
)
use_header_for_delete = st.checkbox("CSV 첫 행을 헤더로 사용하기 (체크 시, 첫 행을 컬럼명으로 간주)", key="col_delete_header_main")
if column_delete_target != "--- 선택하세요 ---":
    orig_key_delete = [k for k, v in st.session_state["file_names"].items() if v == column_delete_target][0]
    df_for_delete = st.session_state["csv_dataframes"][orig_key_delete]
    df_temp_delete = df_for_delete.copy()
    if use_header_for_delete and not df_temp_delete.empty:
        df_temp_delete.columns = df_temp_delete.iloc[0].astype(str)
        df_temp_delete = df_temp_delete[1:].reset_index(drop=True)
    columns_list_delete = [str(col) for col in df_temp_delete.columns.tolist()]
    selected_cols_to_delete = st.multiselect(
        "삭제할 열을 선택하세요", options=columns_list_delete, key="col_delete_multiselect_main"
    )
    if st.button("열 삭제 실행", key="col_delete_run_main"):
        if not selected_cols_to_delete:
            st.warning("최소 한 개 이상의 열을 선택해 주세요.")
        else:
            df_after_delete = df_temp_delete.drop(columns=selected_cols_to_delete, errors="ignore")
            base_name, ext = os.path.splitext(column_delete_target)
            new_file_name_deleted = f"{base_name}_cols_removed{ext}"
            save_to_session_and_download(df_after_delete, new_file_name_deleted)
            st.success(f"선택한 열이 삭제된 파일 '{new_file_name_deleted}'이(가) 생성되었습니다.")

# ------------------------------------------------------------------------------
# K. N개로 분할하기
# ------------------------------------------------------------------------------
st.subheader("9) 파일 분할하기") # 번호 수정
split_target = st.selectbox(
    "분할할 CSV를 선택하세요",
    ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values()),
    key="split_select_main"
)
n_parts = st.number_input("몇 개로 분할할까요?", min_value=2, value=2, step=1, key="split_n_parts_main")
if st.button("파일 분할 실행", key="split_run_main"):
    if split_target == "--- 선택하세요 ---":
        st.error("분할할 파일을 선택해 주세요.")
    else:
        orig_key_split = [k for k, v in st.session_state["file_names"].items() if v == split_target][0]
        df_to_split = st.session_state["csv_dataframes"][orig_key_split]
        total_rows = len(df_to_split)
        if total_rows == 0:
            st.warning("선택한 파일이 비어있어 분할할 수 없습니다.")
        elif n_parts > total_rows:
            st.warning(f"분할 개수({n_parts})가 전체 행 수({total_rows})보다 많습니다. 각 파일에 1행씩, 총 {total_rows}개로 분할합니다.")
            n_parts = total_rows
        if total_rows > 0:
            base_chunk = total_rows // n_parts
            remainder = total_rows % n_parts
            st.write(f"총 행 수: {total_rows}, 분할 개수: {n_parts}")
            st.write(f"각 파트 기본 크기: {base_chunk}, 나머지 분배 파트 수: {remainder}")
            start_idx = 0
            for i in range(n_parts):
                chunk_size = base_chunk + (1 if i < remainder else 0)
                if chunk_size == 0: continue
                end_idx = start_idx + chunk_size
                df_chunk = df_to_split.iloc[start_idx:end_idx].reset_index(drop=True)
                name_part, ext_part = os.path.splitext(split_target)
                part_file_name = f"{name_part}_part{i+1}{ext_part}"
                save_to_session_and_download(df_chunk, part_file_name)
                st.info(f"{part_file_name} : 행 {start_idx+1} ~ {end_idx} (총 {chunk_size}행) 생성 완료")
                start_idx = end_idx

# ------------------------------------------------------------------------------
# L. 사용자 지정 파일 순서 및 매트릭스 출력
# ------------------------------------------------------------------------------
st.subheader("10) 사용자 지정 파일 순서 및 매트릭스 출력") # 번호 수정
all_available_files = list(st.session_state["file_names"].values())
if not all_available_files:
    st.warning("업로드된 파일이 없습니다.")
else:
    st.markdown("**매트릭스에 사용할 파일 순서를 직접 지정해주세요.**")
    default_file_count = min(2, len(all_available_files)) if len(all_available_files) > 0 else 1
    ordered_file_count = st.number_input("매트릭스에 사용할 파일 개수", 
                                           min_value=1, max_value=len(all_available_files), 
                                           value=default_file_count, step=1, key="matrix_file_count_main")
    ordered_files_for_matrix = []
    temp_available_files = list(all_available_files)
    for i in range(int(ordered_file_count)):
        file_sel = st.selectbox(f"{i+1}번째 파일 선택", temp_available_files, key=f"order_matrix_select_{i}") # key 수정
        ordered_files_for_matrix.append(file_sel)
    if len(set(ordered_files_for_matrix)) < len(ordered_files_for_matrix) and ordered_file_count > 0 :
        st.error("같은 파일이 여러 번 선택되었습니다. 각 순서에는 서로 다른 파일을 선택해주세요.")
    else:
        st.markdown("### 출력 형식 선택")
        representation_option = st.radio(
            "어떻게 결과를 표시할까요?", ("절대값 (비율)", "절대값", "비율"), key="representation_option_matrix_main"
        )
        if st.button("매트릭스 생성하기", key="matrix_run_main"):
            uid_sets_matrix = {}
            for fname_matrix in ordered_files_for_matrix:
                orig_key_matrix = [k for k, v in st.session_state["file_names"].items() if v == fname_matrix][0]
                uid_sets_matrix[fname_matrix] = get_uid_set(orig_key_matrix)
            matrix_data = []; header_row = [""] + ordered_files_for_matrix; matrix_data.append(header_row)
            for i, file_i_name in enumerate(ordered_files_for_matrix):
                row_data = [file_i_name]; base_count_i = len(uid_sets_matrix[file_i_name])
                for j, file_j_name in enumerate(ordered_files_for_matrix):
                    if j < i: row_data.append("") 
                    else:
                        inter_count_ij = len(uid_sets_matrix[file_i_name].intersection(uid_sets_matrix[file_j_name]))
                        if representation_option == "절대값": cell_val_str = str(inter_count_ij)
                        elif representation_option == "비율":
                            ratio_val = (round(inter_count_ij / base_count_i * 100, 1) if base_count_i > 0 else 0.0)
                            cell_val_str = f"{ratio_val}%"
                        else:
                            ratio_val = (round(inter_count_ij / base_count_i * 100, 1) if base_count_i > 0 else 0.0)
                            cell_val_str = f"{inter_count_ij} ({ratio_val}%)"
                        row_data.append(cell_val_str)
                matrix_data.append(row_data)
            matrix_df_display = pd.DataFrame(matrix_data[1:], columns=matrix_data[0])
            st.write("### 파일 간 교집합 매트릭스 (i행 파일 기준)")
            st.dataframe(matrix_df_display)
            # save_to_session_and_download 함수는 CSV만 저장하므로, DataFrame을 직접 저장하려면 다른 방식 사용
            # 여기서는 다운로드 버튼만 제공
            csv_buffer_matrix = io.StringIO()
            matrix_df_display.to_csv(csv_buffer_matrix, index=False) # 여기서는 헤더 포함 저장
            st.download_button(label="교집합 매트릭스 다운로드", data=csv_buffer_matrix.getvalue(),
                               file_name="pairwise_intersection_matrix.csv", mime="text/csv",
                               key="download_pairwise_matrix")


            retention_matrix_data = []; header_row_retention = [""] + ordered_files_for_matrix
            retention_matrix_data.append(header_row_retention)
            for i, file_i_ret_name in enumerate(ordered_files_for_matrix):
                row_data_retention = [file_i_ret_name]; base_count_ret_i = len(uid_sets_matrix[file_i_ret_name])
                current_intersection_set = uid_sets_matrix[file_i_ret_name].copy()
                for j, file_j_ret_name in enumerate(ordered_files_for_matrix):
                    if j < i: row_data_retention.append("")  
                    else:
                        if j > i : current_intersection_set = current_intersection_set.intersection(uid_sets_matrix[file_j_ret_name])
                        abs_val_ret = len(current_intersection_set)
                        rate_ret = round(abs_val_ret / base_count_ret_i * 100, 1) if base_count_ret_i > 0 else 0.0
                        if representation_option == "절대값": cell_val_ret_str = str(abs_val_ret)
                        elif representation_option == "비율": cell_val_ret_str = f"{rate_ret}%"
                        else: cell_val_ret_str = f"{abs_val_ret} ({rate_ret}%)"
                        row_data_retention.append(cell_val_ret_str)
                retention_matrix_data.append(row_data_retention)
            retention_df_display = pd.DataFrame(retention_matrix_data[1:], columns=retention_matrix_data[0])
            st.write(f"### 잔존율 매트릭스 ({ordered_files_for_matrix[0] if ordered_files_for_matrix else ''} 기준 시작)") # 빈 리스트 방지
            st.dataframe(retention_df_display)
            # save_to_session_and_download(retention_df_display, "retention_matrix.csv")
            csv_buffer_retention = io.StringIO()
            retention_df_display.to_csv(csv_buffer_retention, index=False) # 헤더 포함 저장
            st.download_button(label="잔존율 매트릭스 다운로드", data=csv_buffer_retention.getvalue(),
                               file_name="retention_matrix.csv", mime="text/csv",
                               key="download_retention_matrix")


# ------------------------------------------------------------------------------
# M. 벤 다이어그램 생성 (matplotlib_venn 임포트 주의)
# ------------------------------------------------------------------------------
st.subheader("11) 벤 다이어그램 생성")
selected_for_venn = st.multiselect(
    "벤 다이어그램에 사용할 CSV 파일 선택 (최대 3개)",
    list(st.session_state["file_names"].values()),
    key="venn_select_main"
)
if st.button("벤 다이어그램 생성하기", key="venn_run_main"):
    if not selected_for_venn:
        st.error("최소 1개의 CSV 파일을 선택해주세요.")
    elif len(selected_for_venn) > 3:
        st.error("벤 다이어그램은 현재 1~3개의 집합만 지원합니다.")
    else:
        try:
            from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
            import matplotlib.patches as mpatches # mpatches도 여기서 임포트
        except ImportError:
            st.error("벤 다이어그램을 생성하려면 'matplotlib-venn' 라이브러리가 필요합니다. `pip install matplotlib-venn`으로 설치해주세요.")
            st.stop()

        venn_sets_dict = {}
        venn_set_labels = []
        for fname_venn in selected_for_venn:
            orig_key_venn = [k for k, v in st.session_state["file_names"].items() if v == fname_venn][0]
            venn_sets_dict[fname_venn] = get_uid_set(orig_key_venn)
            venn_set_labels.append(fname_venn)
        
        fig_venn, ax_venn = plt.subplots(figsize=(8,8))
        plt.title(f"벤 다이어그램 ({len(selected_for_venn)} 집합)", fontsize=14)
        if len(selected_for_venn) == 1:
            set1_venn = venn_sets_dict[venn_set_labels[0]]; count1 = len(set1_venn)
            circle = plt.Circle((0.5, 0.5), 0.4, color="skyblue", alpha=0.7)
            ax_venn.add_patch(circle)
            ax_venn.text(0.5, 0.5, f"{venn_set_labels[0]}\n({count1})", 
                         horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='bold')
            ax_venn.set_xlim(0, 1); ax_venn.set_ylim(0, 1); ax_venn.axis('off')
            legend_patch1 = mpatches.Patch(color='skyblue', label=f"{venn_set_labels[0]} ({count1})")
            ax_venn.legend(handles=[legend_patch1], loc="best", fontsize=10)
        elif len(selected_for_venn) == 2:
            set1_venn = venn_sets_dict[venn_set_labels[0]]; set2_venn = venn_sets_dict[venn_set_labels[1]]
            v = venn2([set1_venn, set2_venn], set_labels=tuple(venn_set_labels), ax=ax_venn,
                      set_colors=('skyblue', 'lightcoral'), alpha = 0.7)
            for text_id in ['10', '01', '11']:
                if v.get_label_by_id(text_id): v.get_label_by_id(text_id).set_fontsize(10)
            venn2_circles([set1_venn, set2_venn], linestyle='dashed', linewidth=1, color='grey', ax=ax_venn)
        elif len(selected_for_venn) == 3:
            set1_venn = venn_sets_dict[venn_set_labels[0]]; set2_venn = venn_sets_dict[venn_set_labels[1]]
            set3_venn = venn_sets_dict[venn_set_labels[2]]
            v = venn3([set1_venn, set2_venn, set3_venn], set_labels=tuple(venn_set_labels), ax=ax_venn,
                      set_colors=('skyblue', 'lightcoral', 'lightgreen'), alpha = 0.7)
            for text_id in ['100', '010', '001', '110', '101', '011', '111']:
                if v.get_label_by_id(text_id): v.get_label_by_id(text_id).set_fontsize(9)
            venn3_circles([set1_venn, set2_venn, set3_venn], linestyle='dashed', linewidth=1, color='grey', ax=ax_venn)
        st.pyplot(fig_venn)


# ------------------------------------------------------------------------------
# N. 카카오페이지 웹툰 업데이트 일자 추출 (이전 답변의 코드 통합)
# ------------------------------------------------------------------------------
st.subheader("12) 카카오페이지 웹툰 업데이트 일자 추출")

kakaopage_series_ids_input = st.text_input(
    "카카오페이지 작품 ID를 쉼표(,)로 구분하여 입력하세요 (예: 59782511, 12345678)",
    key="kakaopage_ids_input_main"
)

@st.cache_data(ttl=3600, show_spinner=False) # 1시간 동안 캐싱, 스피너는 수동으로 제어
def get_update_dates_for_series_cached(series_id_tuple, webdriver_options_dict): # 튜플과 딕셔너리로 캐싱 가능하게
    # WebDriver 옵션을 딕셔너리에서 다시 생성
    options = webdriver.ChromeOptions()
    for arg, value in webdriver_options_dict.get("arguments", {}).items():
        if value is None: # 인자만 있는 경우 (e.g., --headless)
            options.add_argument(arg)
        else: # 값이 있는 경우 (e.g., user-agent="...")
            options.add_argument(f"{arg}={value}")
    for opt_name, opt_value in webdriver_options_dict.get("experimental_options", {}).items():
        options.add_experimental_option(opt_name, opt_value)


    # series_id_tuple에서 실제 series_id를 가져옴 (하나만 처리하므로 첫 번째 요소)
    # 이 함수는 ID 하나씩 처리하도록 설계되어야 캐싱이 효과적
    # 실제로는 get_update_dates_for_series_cached 함수 외부에서 ID 리스트를 순회하며 호출하고,
    # 각 호출 결과를 모아서 ZIP 처리하는 것이 더 일반적임.
    # 여기서는 예시로 하나의 ID만 받는다고 가정하고, 실제 사용시는 아래 실행 버튼 로직에서 반복 호출.
    # -> 아래 실행 로직에서 반복 호출하도록 수정. 이 함수는 단일 ID에 대한 처리만 담당.
    
    series_id_single = series_id_tuple # 함수 호출 시 단일 ID를 튜플로 감싸서 전달 가정

    driver_instance = None # finally에서 quit하기 위해 정의
    try:
        # webdriver_manager 사용 권장 (Streamlit Cloud 등 배포 환경에서 유리)
        s = ChromeService(ChromeDriverManager().install())
        driver_instance = webdriver.Chrome(service=s, options=options)
        
        # 실제 스크래핑 함수 호출 (이전 답변의 get_update_dates_for_series 함수)
        # 이 함수는 driver를 인자로 받아야 함
        dates = get_update_dates_for_series_internal(series_id_single, driver_instance) # 내부 함수 호출
        return dates
    except Exception as e:
        # st.error(f"ID {series_id_single} 스크래핑 중 오류(캐시된 함수 내): {e}") # Streamlit 컨텍스트가 아님
        print(f"Error in cached function for {series_id_single}: {e}") # 서버 로그용
        return [] # 오류 시 빈 리스트 반환
    finally:
        if driver_instance:
            driver_instance.quit()


# 실제 스크래핑 로직을 담당하는 내부 함수
def get_update_dates_for_series_internal(series_id, driver):
    url = f"https://page.kakao.com/content/{series_id}"
    driver.get(url)
    update_dates = []

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]"))
        )
        time.sleep(2) # 초기 렌더링 시간 확보

        max_scroll_attempts = 15 # 최대 더보기 시도 횟수 (페이지마다 다를 수 있음)
        no_new_content_streak = 0
        max_no_new_content_streak = 3 # 연속으로 새 내용 없으면 중단
        
        for attempt in range(max_scroll_attempts):
            # 현재 로드된 날짜 수 (스크롤/클릭 전)
            # 날짜 XPath는 <span class="break-all align-middle">YY.MM.DD</span>
            # 더 정확하게는 회차 리스트 아이템 내의 날짜 span
            # .//div[contains(@class, 'font-x-small1') ...]/span[@class='break-all align-middle'][1]
            dates_before_action = driver.find_elements(By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]/li//div[contains(@class, 'font-x-small1')]//span[@class='break-all align-middle'][1]")
            count_before_action = len(dates_before_action)

            try:
                # "더보기" 버튼 (아래 화살표 이미지) 찾기
                load_more_button_xpath = "//ul[contains(@class, 'jsx-3287026398')]/following-sibling::div[1][.//img[@alt='아래 화살표']]"
                load_more_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, load_more_button_xpath))
                )
                # 버튼이 화면에 보이도록 스크롤 (클릭 가로채기 방지)
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", load_more_button)
                time.sleep(0.5) # 스크롤 후 안정화 시간
                
                # JavaScript로 클릭 (가로채기 문제 회피에 도움)
                driver.execute_script("arguments[0].click();", load_more_button)
                # print(f"ID {series_id}: '더보기' 버튼 클릭 시도 ({attempt + 1}/{max_scroll_attempts})") # 디버깅용
                time.sleep(1.5) # 새 콘텐츠 로드 대기 시간 (네트워크 상태에 따라 조절)

                # 클릭 후 날짜 수 확인
                dates_after_action = driver.find_elements(By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]/li//div[contains(@class, 'font-x-small1')]//span[@class='break-all align-middle'][1]")
                count_after_action = len(dates_after_action)

                if count_after_action == count_before_action:
                    no_new_content_streak += 1
                    # print(f"ID {series_id}: 새 콘텐츠 없음 ({no_new_content_streak}/{max_no_new_content_streak})") # 디버깅용
                    if no_new_content_streak >= max_no_new_content_streak:
                        # print(f"ID {series_id}: 연속 {max_no_new_content_streak}회 새 콘텐츠 없어 더보기 중단.")
                        break 
                else:
                    no_new_content_streak = 0 # 새 콘텐츠 있으면 초기화
            
            except TimeoutException:
                # print(f"ID {series_id}: '더보기' 버튼 타임아웃. 모든 회차 로드 완료로 간주.")
                break
            except NoSuchElementException:
                # print(f"ID {series_id}: '더보기' 버튼 없음. 모든 회차 로드 완료로 간주.")
                break
            except ElementClickInterceptedException:
                # print(f"ID {series_id}: '더보기' 버튼 클릭 가로채짐. 페이지 하단으로 스크롤 후 재시도.")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1) # 스크롤 후 대기
            except Exception as e_click:
                # print(f"ID {series_id}: '더보기' 버튼 클릭 중 기타 오류: {e_click}")
                break # 예측 못한 오류 시 중단

        # 모든 "더보기" 완료 후 최종 날짜 추출
        episode_list_items_xpath = "//ul[contains(@class, 'jsx-3287026398')]/li[contains(@class, 'list-child-item')]"
        list_items = driver.find_elements(By.XPATH, episode_list_items_xpath)
        
        date_pattern = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")
        extracted_this_run = []
        for item in list_items:
            try:
                # 각 li 아이템 내부에서 날짜 정보를 포함하는 span 찾기
                # <span class="break-all align-middle">YY.MM.DD</span>
                # 이 span은 div class="line-clamp-1 text-ellipsis font-x-small1 h-16pxr text-el-50" 안에 있음
                date_span_xpath = ".//div[contains(@class, 'font-x-small1') and contains(@class, 'h-16pxr') and contains(@class, 'text-el-50')]/span[@class='break-all align-middle'][1]"
                date_span = item.find_element(By.XPATH, date_span_xpath)
                date_text = date_span.text.strip()
                
                if date_pattern.match(date_text):
                    extracted_this_run.append(date_text)
            except NoSuchElementException:
                pass # 해당 li에 날짜 형식이 없거나 구조가 다를 수 있음
            except Exception as e_item_extract:
                # print(f"ID {series_id}: 개별 회차 날짜 추출 중 오류: {e_item_extract}")
                pass
        
        # 중복 제거 및 순서 유지 (페이지에 나타난 순, 보통 최신순)
        seen_dates = set()
        for d_item in extracted_this_run:
            if d_item not in seen_dates:
                update_dates.append(d_item)
                seen_dates.add(d_item)
                
    except TimeoutException:
        # print(f"ID {series_id}: 페이지 로드 시간 초과 (WebDriverWait)")
        st.warning(f"ID {series_id}: 페이지 로드 시간 초과 또는 주요 요소를 찾을 수 없습니다.")
    except Exception as e_global_scrape:
        # print(f"ID {series_id}: 스크래핑 중 전역 오류: {e_global_scrape}")
        st.error(f"ID {series_id}: 스크래핑 과정 중 오류 발생: {e_global_scrape}")
    
    return update_dates


if st.button("업데이트 일자 추출 및 ZIP 다운로드", key="kakaopage_extract_button_main"):
    if not kakaopage_series_ids_input:
        st.warning("작품 ID를 입력해주세요.")
    else:
        series_ids_list = [id_str.strip() for id_str in kakaopage_series_ids_input.split(',') if id_str.strip()]
        if not series_ids_list:
            st.warning("유효한 작품 ID가 없습니다.")
        else:
            # WebDriver 옵션 설정 (캐시 함수에 전달하기 위해 딕셔너리 형태로)
            webdriver_options_dict = {
                "arguments": {
                    "--headless": None,
                    "--no-sandbox": None,
                    "--disable-dev-shm-usage": None,
                    "--disable-gpu": None,
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36", # User-Agent는 최신으로 유지 권장
                    "--lang": "ko_KR"
                },
                "experimental_options": {
                    "excludeSwitches": ['enable-logging']
                }
            }
            
            results_for_zip = {}
            total_ids = len(series_ids_list)
            
            # UI 업데이트를 위한 st 요소들
            progress_bar_placeholder = st.empty()
            status_text_placeholder = st.empty()
            
            with st.spinner("작품 정보를 처리 중입니다..."): # 전체 작업에 대한 스피너
                progress_bar = progress_bar_placeholder.progress(0)
                
                for i, series_id_item in enumerate(series_ids_list):
                    current_progress = (i + 1) / total_ids
                    progress_bar.progress(current_progress)
                    status_text_placeholder.text(f"처리 중: {series_id_item} ({i+1}/{total_ids})")
                    
                    try:
                        # 캐시된 함수 호출 시 series_id를 튜플로 감싸서 전달 (캐시 키로 사용 위함)
                        # 하지만 단일 ID 처리이므로 그냥 문자열로 전달해도 내부에서 처리 가능
                        # 여기서는 get_update_dates_for_series_cached가 series_id_tuple을 받고,
                        # 그 안에서 첫번째 요소를 실제 ID로 사용한다고 가정.
                        # 더 명확하게는 캐시 함수가 단일 ID만 받도록 하고, 외부에서 루프 돌리는게 좋음.
                        # 위 get_update_dates_for_series_cached 함수는 단일 ID를 받도록 수정했으므로 그대로 전달.
                        dates = get_update_dates_for_series_cached(series_id_item, webdriver_options_dict)
                        
                        if dates:
                            file_content = "\n".join(dates)
                            safe_series_id = re.sub(r'[\\/*?:"<>|]', "_", series_id_item)
                            filename_in_zip = f"{safe_series_id}_updates.txt"
                            results_for_zip[filename_in_zip] = file_content
                            st.info(f"'{series_id_item}' 작품: {len(dates)}개의 업데이트 일자 추출 완료.")
                        else:
                            st.warning(f"'{series_id_item}' 작품: 업데이트 일자를 찾을 수 없거나 추출에 실패했습니다.")
                    except Exception as e_scrape_loop:
                        st.error(f"'{series_id_item}' 작품 처리 중 오류 발생: {e_scrape_loop}")
                    
                    time.sleep(0.5) # 개별 작업 후 약간의 텀 (UI 업데이트 및 서버 부하 고려)

            # 모든 ID 처리 후
            progress_bar_placeholder.empty() # 프로그레스 바 제거
            status_text_placeholder.empty() # 상태 메시지 제거

            if results_for_zip:
                st.info("모든 작품 처리 완료. ZIP 파일 생성 중...")
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for filename, content in results_for_zip.items():
                        zf.writestr(filename, content.encode('utf-8')) # 문자열을 바이트로 인코딩
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="추출된 업데이트 일자 ZIP 다운로드",
                    data=zip_buffer,
                    file_name="kakaopage_webtoon_updates.zip", # 파일명 구체화
                    mime="application/zip",
                    key="download_kakaopage_zip_main"
                )
                st.success("ZIP 파일 생성 완료! 위 버튼으로 다운로드하세요.")
            else:
                st.warning("추출된 데이터가 없어 ZIP 파일을 생성할 수 없습니다.")
