import math
import streamlit as st
import pandas as pd
import io
import zipfile
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles # 벤다이어그램 사용 시 (필요시 설치)

# Selenium 관련 import
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import re
import os # 파일명 처리 등

st.set_page_config(page_title="Pinsight CSV & Webtoon Utility", layout="wide") # 앱 제목 변경
st.title("Pinsight CSV & 웹툰 정보 추출 도구") # 앱 제목 변경

# ------------------------------------------------------------------------------
# A. CSV 업로드 처리
# ------------------------------------------------------------------------------
st.header("📄 CSV 파일 관리") # 헤더로 변경
st.subheader("1) CSV 업로드")

uploaded_files = st.file_uploader(
    "여기에 CSV 파일들을 드래그하거나, 'Browse files' 버튼을 눌러 선택하세요. (각 CSV는 첫 번째 열에 ID가 있어야 합니다)",
    type=["csv"],
    accept_multiple_files=True
)

# 세션 상태 초기화 (앱 실행 시 한 번만)
if "csv_dataframes" not in st.session_state:
    st.session_state["csv_dataframes"] = {}
if "file_names" not in st.session_state:
    st.session_state["file_names"] = {}

if st.button("CSV 로드하기", key="load_csv_button"):
    if uploaded_files:
        loaded_count = 0
        for uploaded_file in uploaded_files:
            # 이미 로드된 파일명인지 확인 (덮어쓰기 방지 또는 사용자에게 알림)
            if uploaded_file.name in st.session_state["csv_dataframes"]:
                st.warning(f"'{uploaded_file.name}' 파일은 이미 로드되어 있습니다. 새로 로드하려면 기존 파일을 삭제 후 진행해주세요.", icon="⚠️")
                continue # 다음 파일로 넘어감

            try:
                # CSV를 읽을 때, 첫 번째 열만 사용하고 나머지는 무시하거나, 필요에 따라 처리
                # 여기서는 모든 열을 읽되, UID는 첫 번째 열로 가정
                df = pd.read_csv(uploaded_file, header=None, dtype=str) # 모든 데이터를 문자열로 읽기 (UID 일관성)
                if df.empty or df.shape[1] == 0:
                    st.error(f"'{uploaded_file.name}' 파일이 비어있거나 유효한 데이터가 없습니다.")
                    continue

                st.session_state["csv_dataframes"][uploaded_file.name] = df
                st.session_state["file_names"][uploaded_file.name] = uploaded_file.name # 초기 파일명 설정
                st.success(f"'{uploaded_file.name}': 업로드 & 로드 성공 (행 수: {len(df)})")
                loaded_count +=1
            except pd.errors.EmptyDataError:
                st.error(f"'{uploaded_file.name}': 파일이 비어있어 로드할 수 없습니다.")
            except Exception as e:
                st.error(f"'{uploaded_file.name}': 로드 실패 - {e}")
        if loaded_count > 0 :
            st.rerun() # 파일 목록 즉시 업데이트
    else:
        st.warning("업로드된 파일이 없습니다.")

# ------------------------------------------------------------------------------
# B. 업로드된 파일 목록 표시 (파일명 변경 및 삭제 기능)
# ------------------------------------------------------------------------------
if st.session_state["csv_dataframes"]:
    st.write("### 업로드된 파일 목록 및 관리")
    
    # 삭제할 파일 키를 임시 저장 (순회 중 딕셔셔너리 변경 방지)
    keys_to_delete_from_session = []

    for original_name in list(st.session_state["csv_dataframes"].keys()): # 순회 중 변경 방지
        if original_name not in st.session_state["csv_dataframes"]: continue # 이미 삭제된 경우

        df_display = st.session_state["csv_dataframes"][original_name]
        
        col_name, col_info, col_download, col_delete = st.columns([3, 2, 1.5, 1])

        with col_name:
            # 파일명 변경 입력 필드
            current_display_name = st.session_state["file_names"].get(original_name, original_name)
            new_display_name = st.text_input(
                f"파일명 (원본: {original_name})",
                value=current_display_name,
                key=f"text_input_filename_{original_name}"
            )
            if new_display_name != current_display_name:
                # 새 파일명이 다른 파일의 현재 표시명과 중복되는지 확인
                other_display_names = [
                    name for key, name in st.session_state["file_names"].items() if key != original_name
                ]
                if new_display_name in other_display_names:
                    st.warning(f"표시명 '{new_display_name}'이(가) 이미 다른 파일에 사용 중입니다.", icon="⚠️")
                else:
                    st.session_state["file_names"][original_name] = new_display_name
                    st.rerun() # 파일명 변경 시 UI 즉시 업데이트

        with col_info:
            st.write(f"행 수: {len(df_display)}")
            # 첫 번째 열의 UID 개수 (중복 제거)
            if not df_display.empty and df_display.shape[1] > 0:
                unique_uids = df_display.iloc[:, 0].nunique()
                st.write(f"고유 ID 수 (첫 열): {unique_uids}")
            else:
                st.write("고유 ID 수 (첫 열): 데이터 없음")


        with col_download:
            csv_buffer_download = io.StringIO()
            # 다운로드 시에는 원본 데이터프레임을 사용
            st.session_state["csv_dataframes"][original_name].to_csv(csv_buffer_download, index=False, header=False)
            st.download_button(
                label="CSV 다운로드",
                data=csv_buffer_download.getvalue(),
                file_name=st.session_state["file_names"].get(original_name, original_name), # 현재 표시명으로 다운로드
                mime="text/csv",
                key=f"download_btn_individual_{original_name}"
            )

        with col_delete:
            if st.button("파일 삭제", key=f"delete_btn_individual_{original_name}"):
                keys_to_delete_from_session.append(original_name)
    
    if keys_to_delete_from_session:
        for key_del in keys_to_delete_from_session:
            if key_del in st.session_state["csv_dataframes"]:
                del st.session_state["csv_dataframes"][key_del]
            if key_del in st.session_state["file_names"]:
                del st.session_state["file_names"][key_del]
        st.success(f"{len(keys_to_delete_from_session)}개 파일이 세션에서 삭제되었습니다.")
        st.rerun() # 삭제 후 UI 갱신

st.markdown("---") # CSV 관리 섹션 끝

# ------------------------------------------------------------------------------
# C. 공통 함수 (CSV 조작용)
# ------------------------------------------------------------------------------
def get_uid_set_from_display_name(display_name):
    """표시 파일명으로 원본 키를 찾아 UID set 생성"""
    original_key = next((k for k, v in st.session_state["file_names"].items() if v == display_name), None)
    if original_key and original_key in st.session_state["csv_dataframes"]:
        df = st.session_state["csv_dataframes"][original_key]
        if not df.empty and df.shape[1] > 0:
            return set(df.iloc[:, 0].astype(str).dropna().unique()) # NaN 제거 및 고유값
    return set() # 파일을 찾지 못하거나 비어있으면 빈 set 반환

def save_result_to_session_and_offer_download(result_df, base_filename="result.csv"):
    """결과 DataFrame을 새 파일로 세션에 추가하고 다운로드 버튼 제공"""
    unique_name_candidate = base_filename
    counter = 1
    # 현재 사용 중인 모든 표시 파일명 목록
    all_current_display_names = list(st.session_state["file_names"].values())

    # 새 파일명이 기존 표시 파일명과 충돌하지 않도록 조정
    while unique_name_candidate in all_current_display_names:
        name_part, ext_part = os.path.splitext(base_filename)
        unique_name_candidate = f"{name_part}_{counter}{ext_part}"
        counter += 1
    
    # 새 파일에 대한 원본 키는 표시명과 동일하게 사용 (단순화)
    # 또는 고유 ID (e.g., timestamp)를 생성하여 사용할 수도 있음
    new_original_key = unique_name_candidate 
    
    st.session_state["csv_dataframes"][new_original_key] = result_df
    st.session_state["file_names"][new_original_key] = unique_name_candidate # 표시명도 동일하게 설정

    csv_buffer_result = io.StringIO()
    result_df.to_csv(csv_buffer_result, index=False, header=False) # UID 목록이므로 헤더 없이 저장
    
    st.download_button(
        label=f"'{unique_name_candidate}' 다운로드 ({len(result_df)}개 ID)",
        data=csv_buffer_result.getvalue(),
        file_name=unique_name_candidate,
        mime="text/csv",
        key=f"download_generated_csv_{unique_name_candidate.replace('.', '_')}" # 고유 키
    )
    st.success(f"결과 파일 '{unique_name_candidate}'이(가) 생성되어 목록에 추가되었습니다.")
    st.rerun() # 새 파일 목록 즉시 반영

# ------------------------------------------------------------------------------
# D. 교집합 ~ M. 벤 다이어그램 (기존 CSV 기능들)
# ------------------------------------------------------------------------------
# (이전 답변의 D부터 M까지의 CSV 조작 기능 코드를 여기에 그대로 붙여넣으세요.)
# 각 st.multiselect, st.selectbox 등의 key가 중복되지 않도록 주의하세요.
# 함수 호출 시 get_uid_set 대신 get_uid_set_from_display_name 사용.
# 결과 저장 시 save_to_session_and_download 대신 save_result_to_session_and_offer_download 사용.

# 예시: 교집합 기능 수정
st.header("🛠️ CSV 파일 조합 및 분석") # 헤더
st.subheader("2) 교집합 (Intersection)")
selected_for_intersect = st.multiselect(
    "교집합 대상 CSV 선택 (2개 이상)",
    list(st.session_state["file_names"].values()), # 표시명으로 선택
    key="intersect_select_main"
)
if st.button("교집합 실행하기", key="intersect_run_main"):
    if len(selected_for_intersect) < 2:
        st.error("교집합은 2개 이상 선택해야 합니다.")
    else:
        base_set_intersect = None
        for display_name_intersect in selected_for_intersect:
            current_set_intersect = get_uid_set_from_display_name(display_name_intersect)
            if base_set_intersect is None:
                base_set_intersect = current_set_intersect
            else:
                base_set_intersect.intersection_update(current_set_intersect) # intersection_update 사용

        if base_set_intersect is not None:
            st.write(f"교집합 결과 UID 수: {len(base_set_intersect)}")
            result_df_intersect = pd.DataFrame(sorted(list(base_set_intersect))) # 정렬된 DataFrame
            save_result_to_session_and_offer_download(result_df_intersect, "result_intersection.csv")
        else:
            st.warning("교집합을 계산할 수 없습니다 (선택된 파일에 데이터가 없거나 문제가 있을 수 있습니다).")

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


st.markdown("---") # CSV 기능과 웹툰 추출 기능 구분

# ... (기존 Streamlit 앱 코드 상단은 동일) ...
# ... (A부터 M까지의 CSV 관련 기능 코드는 여기에 그대로 있다고 가정) ...

# ... (기존 Streamlit 앱 코드 상단은 동일) ...
# ... (A부터 M까지의 CSV 관련 기능 코드는 여기에 그대로 있다고 가정) ...

# ... (기존 Streamlit 앱 코드 상단은 동일) ...
# ... (A부터 M까지의 CSV 관련 기능 코드는 여기에 그대로 있다고 가정) ...

# ------------------------------------------------------------------------------
# N. 카카오페이지 웹툰 업데이트 일자 추출 (XPath 수정 및 상세 로그 강화)
# ------------------------------------------------------------------------------
st.header("🌐 카카오페이지 웹툰 정보 추출")
st.subheader("업데이트 일자 추출")

kakaopage_series_ids_input_kp = st.text_input(
    "카카오페이지 작품 ID를 쉼표(,)로 구분하여 입력하세요 (예: 59782511, 12345678)",
    key="kakaopage_ids_input_main_kp_v6" # 키 변경
)

log_container_kp_v6 = st.container()
process_logs_kp_v6 = []

# 실제 스크래핑 로직을 담당하는 내부 함수
def get_update_dates_for_series_internal_v6(series_id, driver, log_callback_ui_v6):
    url = f"https://page.kakao.com/content/{series_id}"
    log_callback_ui_v6(f"ID {series_id}: 스크래핑 시작. URL: {url}")
    driver.get(url)
    update_dates = []
    
    try:
        WebDriverWait(driver, 20).until( 
            EC.presence_of_element_located((By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]"))
        )
        log_callback_ui_v6(f"ID {series_id}: 초기 회차 목록 컨테이너(ul) 로드 확인.")
        time.sleep(3.5) # JavaScript 및 동적 콘텐츠 로드를 위한 충분한 초기 대기 시간 (조금 더 늘림)

        max_load_more_clicks = 35 # 더보기 버튼 클릭 최대 시도 횟수 (충분히 크게)
        no_new_content_streak = 0
        max_no_new_content_streak = 4 # 4번 연속 새 회차 로드 안되면 중단 (네트워크 지연 등 고려)
        last_known_episode_elements_count = 0

        initial_items_count = len(driver.find_elements(By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]/li"))
        log_callback_ui_v6(f"ID {series_id}: '더보기' 전, 초기 감지된 회차 아이템 수: {initial_items_count}")

        for click_attempt in range(max_load_more_clicks):
            try:
                # 현재 화면에 있는 모든 회차의 날짜 요소들을 가져옴 (새 콘텐츠 감지용)
                current_episode_date_elements = driver.find_elements(By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]/li//div[contains(@class, 'font-x-small1')]//span[@class='break-all align-middle'][1]")
                current_elements_count = len(current_episode_date_elements)
                log_callback_ui_v6(f"ID {series_id}: '더보기' 시도 {click_attempt + 1}/{max_load_more_clicks}. 현재 감지된 날짜 요소 수: {current_elements_count}.")

                if current_elements_count == last_known_episode_elements_count and click_attempt > 0 :
                    no_new_content_streak += 1
                    log_callback_ui_v6(f"ID {series_id}: 새 날짜 요소 변화 없음 ({no_new_content_streak}/{max_no_new_content_streak}).")
                    if no_new_content_streak >= max_no_new_content_streak:
                        log_callback_ui_v6(f"ID {series_id}: 연속 {max_no_new_content_streak}회 새 날짜 요소 변화 없어 '더보기' 중단.")
                        break
                else:
                    no_new_content_streak = 0
                last_known_episode_elements_count = current_elements_count

                # --- "더보기" 버튼 XPath 수정 (제공해주신 Outer HTML 기반) ---
                load_more_button_xpath = "//div[contains(@class, 'cursor-pointer') and .//img[@alt='아래 화살표']]"
                # --- XPath 수정 끝 ---
                
                log_callback_ui_v6(f"ID {series_id}: 사용된 '더보기' 버튼 XPath: {load_more_button_xpath}")
                
                # 버튼이 DOM에 나타나고 클릭 가능할 때까지 명시적으로 대기
                load_more_button = WebDriverWait(driver, 15).until( # 대기 시간 15초
                    EC.element_to_be_clickable((By.XPATH, load_more_button_xpath))
                )
                log_callback_ui_v6(f"ID {series_id}: '더보기' 버튼 찾음 및 클릭 가능 상태 확인.")
                
                # 버튼이 화면 중앙에 오도록 스크롤 (클릭 정확도 향상 및 가려짐 방지)
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", load_more_button)
                time.sleep(0.8) # 스크롤 후 버튼이 안정화될 시간

                driver.execute_script("arguments[0].click();", load_more_button)
                log_callback_ui_v6(f"ID {series_id}: '더보기' 버튼 클릭 성공 ({click_attempt + 1}).")
                
                time.sleep(3.0) # 새 콘텐츠 로드 대기 시간 (충분히)

            except TimeoutException:
                log_callback_ui_v6(f"ID {series_id}: '더보기' 버튼을 시간 내에 찾거나 클릭할 수 없음 (Timeout). 모든 회차 로드 완료로 간주.")
                break 
            except NoSuchElementException:
                log_callback_ui_v6(f"ID {series_id}: '더보기' 버튼을 더 이상 찾을 수 없음 (NoSuchElement). 모든 회차 로드 완료로 간주.")
                break
            except ElementClickInterceptedException:
                log_callback_ui_v6(f"ID {series_id}: '더보기' 버튼 클릭이 다른 요소에 의해 가로채짐. 페이지 맨 아래로 스크롤 후 재시도.")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.8) # 스크롤 후 안정화 및 재시도 대기 (조금 더 늘림)
            except Exception as e_load_more_v6:
                log_callback_ui_v6(f"ID {series_id}: '더보기' 과정 중 예상치 못한 오류: {str(e_load_more_v6)[:120]}. '더보기' 중단.") # 오류 메시지 길이 조절
                break
        
        log_callback_ui_v6(f"ID {series_id}: '더보기' 과정 완료. 최종 날짜 추출 시작.")

        episode_list_item_xpath_final = "//ul[contains(@class, 'jsx-3287026398')]/li[contains(@class, 'list-child-item')]"
        list_item_elements_final = driver.find_elements(By.XPATH, episode_list_item_xpath_final)
        log_callback_ui_v6(f"ID {series_id}: 최종적으로 스캔할 회차 아이템 li 요소 수: {len(list_item_elements_final)}.")
        
        if not list_item_elements_final:
            log_callback_ui_v6(f"ID {series_id}: [경고] 최종 회차 아이템 li 요소를 하나도 찾지 못했습니다.")

        date_pattern = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")
        all_extracted_dates_in_order = []

        for idx, item_element_final in enumerate(list_item_elements_final):
            try:
                date_span_xpath_final = ".//div[contains(@class, 'font-x-small1') and contains(@class, 'text-el-50')]/span[@class='break-all align-middle'][1]"
                date_span_element = item_element_final.find_element(By.XPATH, date_span_xpath_final)
                date_text = date_span_element.text.strip()
                
                if date_pattern.match(date_text):
                    all_extracted_dates_in_order.append(date_text)
            except NoSuchElementException:
                pass 
            except Exception: 
                pass
        
        log_callback_ui_v6(f"ID {series_id}: 총 {len(all_extracted_dates_in_order)}개의 날짜 텍스트 추출 (중복 포함).")

        seen_dates = set()
        for d_item_final_v6 in all_extracted_dates_in_order: # 변수명 충돌 방지
            if d_item_final_v6 not in seen_dates:
                update_dates.append(d_item_final_v6)
                seen_dates.add(d_item_final_v6)
        
        if not update_dates:
            log_callback_ui_v6(f"ID {series_id}: [결과] 최종적으로 추출된 고유 업데이트 날짜가 없습니다.")
        else:
            log_callback_ui_v6(f"ID {series_id}: [결과] {len(update_dates)}개의 고유한 업데이트 날짜 추출 완료.")
                
    except TimeoutException:
        log_callback_ui_v6(f"ID {series_id}: [오류] 페이지의 주요 컨텐츠(회차 목록) 로드 시간 초과.")
    except Exception as e_global_scrape_final_v6:
        log_callback_ui_v6(f"ID {series_id}: [오류] 스크래핑 중 예기치 않은 전역 오류 발생: {str(e_global_scrape_final_v6)[:150]}")
    
    return update_dates

@st.cache_data(ttl=3600, show_spinner=False)
def get_update_dates_for_series_cached_wrapper_v6(series_id, webdriver_options_dict): # 변수명 변경
    temp_logs_for_cache_v6 = []
    def append_log_for_cache_internal_v6(message):
        temp_logs_for_cache_v6.append(message)

    options = webdriver.ChromeOptions()
    for arg_name, arg_val in webdriver_options_dict.get("arguments", {}).items():
        if arg_val is None: options.add_argument(arg_name)
        else: options.add_argument(f"{arg_name}={arg_val}")
    for opt_name, opt_value in webdriver_options_dict.get("experimental_options", {}).items():
        options.add_experimental_option(opt_name, opt_value)

    driver_instance_cache_internal_v6 = None
    try:
        # 시스템 PATH의 ChromeDriver 사용
        driver_instance_cache_internal_v6 = webdriver.Chrome(options=options)
        append_log_for_cache_internal_v6(f"ID {series_id}: 시스템 PATH의 ChromeDriver로 세션 생성 시도.")
        
        dates = get_update_dates_for_series_internal_v6(series_id, driver_instance_cache_internal_v6, append_log_for_cache_internal_v6)
        return dates, temp_logs_for_cache_v6
    except Exception as e_cache_internal_v6:
        append_log_for_cache_internal_v6(f"ID {series_id}: 캐시된 WebDriver 실행 중 심각한 오류: {str(e_cache_internal_v6)}")
        import traceback
        append_log_for_cache_internal_v6(f"Traceback: {traceback.format_exc(limit=5)}")
        return [], temp_logs_for_cache_v6
    finally:
        if driver_instance_cache_internal_v6:
            driver_instance_cache_internal_v6.quit()
            append_log_for_cache_internal_v6(f"ID {series_id}: WebDriver 세션 종료됨.")

if st.button("업데이트 일자 추출 및 ZIP 다운로드", key="kakaopage_extract_button_main_kp_v6"): # 키 변경
    if not kakaopage_series_ids_input_kp:
        st.warning("작품 ID를 입력해주세요.")
    else:
        series_ids_list_kp_v6 = [id_str.strip() for id_str in kakaopage_series_ids_input_kp.split(',') if id_str.strip()]
        if not series_ids_list_kp_v6:
            st.warning("유효한 작품 ID가 없습니다.")
        else:
            webdriver_options_dict_for_cache_final_kp_v6 = {
                "arguments": {
                    "--headless": None, "--no-sandbox": None, "--disable-dev-shm-usage": None,
                    "--disable-gpu": None,
                    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "--lang": "ko_KR",
                    "--window-size": "1920,1080"
                },
                "experimental_options": {"excludeSwitches": ['enable-logging']}
            }
            
            results_for_zip_kp_v6 = {}
            total_ids_kp_v6 = len(series_ids_list_kp_v6)
            
            process_logs_kp_v6.clear()
            log_container_kp_v6.empty()
            log_display_area_kp_v6 = log_container_kp_v6.expander("실시간 처리 로그 보기", expanded=True)
            
            progress_bar_placeholder_kp_v6 = st.empty()
            current_status_text_kp_v6 = st.empty()
            overall_start_time_kp_v6 = time.time()

            with st.spinner(f"총 {total_ids_kp_v6}개의 작품 정보를 처리 중입니다..."):
                progress_bar_kp_v6 = progress_bar_placeholder_kp_v6.progress(0)
                
                for i, series_id_item_kp_v6 in enumerate(series_ids_list_kp_v6):
                    start_time_item_kp_v6 = time.time()
                    current_progress_kp_v6 = (i + 1) / total_ids_kp_v6
                    progress_bar_kp_v6.progress(current_progress_kp_v6)
                    current_status_text_kp_v6.text(f"처리 중: {series_id_item_kp_v6} ({i+1}/{total_ids_kp_v6})")
                    
                    process_logs_kp_v6.append(f"--- ID: {series_id_item_kp_v6} 처리 시작 ---")
                    with log_display_area_kp_v6: st.markdown(f"**ID: {series_id_item_kp_v6} 처리 시작...**")
                    
                    dates_kp_v6, item_logs_from_cache_kp_v6 = get_update_dates_for_series_cached_wrapper_v6(series_id_item_kp_v6, webdriver_options_dict_for_cache_final_kp_v6)
                    
                    process_logs_kp_v6.extend(item_logs_from_cache_kp_v6)
                    with log_display_area_kp_v6:
                        for log_msg_kp_v6 in item_logs_from_cache_kp_v6:
                            if "[오류]" in log_msg_kp_v6 or "[경고]" in log_msg_kp_v6 or "오류:" in log_msg_kp_v6: st.warning(log_msg_kp_v6)
                            else: st.info(log_msg_kp_v6)
                    
                    if dates_kp_v6:
                        file_content_kp_v6 = "\n".join(dates_kp_v6)
                        safe_series_id_kp_v6 = re.sub(r'[\\/*?:"<>|]', "_", series_id_item_kp_v6)
                        filename_in_zip_kp_v6 = f"{safe_series_id_kp_v6}_updates.txt"
                        results_for_zip_kp_v6[filename_in_zip_kp_v6] = file_content_kp_v6
                        final_msg_kp_v6 = f"ID {series_id_item_kp_v6}: [성공] {len(dates_kp_v6)}개 업데이트 일자 추출 완료."
                        process_logs_kp_v6.append(final_msg_kp_v6)
                        with log_display_area_kp_v6: st.success(final_msg_kp_v6)
                    else:
                        final_msg_kp_v6 = f"ID {series_id_item_kp_v6}: [실패] 업데이트 일자를 찾을 수 없거나 추출에 실패했습니다."
                        process_logs_kp_v6.append(final_msg_kp_v6)
                        with log_display_area_kp_v6: st.error(final_msg_kp_v6)
                    
                    end_time_item_kp_v6 = time.time()
                    process_logs_kp_v6.append(f"ID {series_id_item_kp_v6} 처리 소요 시간: {end_time_item_kp_v6 - start_time_item_kp_v6:.2f}초")
                    process_logs_kp_v6.append(f"--- ID: {series_id_item_kp_v6} 처리 종료 ---\n")
                    with log_display_area_kp_v6: st.markdown("---")
                    time.sleep(0.3)

            progress_bar_placeholder_kp_v6.empty()
            current_status_text_kp_v6.empty()
            overall_end_time_kp_v6 = time.time()
            process_logs_kp_v6.insert(0, f"**카카오페이지 추출 전체 작업 완료! 총 소요 시간: {overall_end_time_kp_v6 - overall_start_time_kp_v6:.2f}초**")

            if results_for_zip_kp_v6:
                log_final_summary_kp_v6 = "모든 작품 처리 완료. ZIP 파일 생성 중..."
                process_logs_kp_v6.append(log_final_summary_kp_v6)

                zip_buffer_kp_v6 = io.BytesIO()
                with zipfile.ZipFile(zip_buffer_kp_v6, "w", zipfile.ZIP_DEFLATED) as zf_kp_v6:
                    for filename_kp_v6, content_kp_v6 in results_for_zip_kp_v6.items():
                        zf_kp_v6.writestr(filename_kp_v6, content_kp_v6.encode('utf-8'))
                zip_buffer_kp_v6.seek(0)
                
                st.download_button(
                    label="추출된 업데이트 일자 ZIP 다운로드",
                    data=zip_buffer_kp_v6,
                    file_name="kakaopage_webtoon_updates.zip",
                    mime="application/zip",
                    key="download_kakaopage_zip_main_v6" # 키 변경
                )
                process_logs_kp_v6.append("ZIP 파일 생성 완료! 위 버튼으로 다운로드하세요.")
                st.success("카카오페이지 업데이트 일자 ZIP 파일 생성 완료! 다운로드 버튼을 이용하세요.")
            else:
                log_final_summary_kp_v6 = "추출된 데이터가 없어 ZIP 파일을 생성할 수 없습니다."
                process_logs_kp_v6.append(log_final_summary_kp_v6)
                st.warning(log_final_summary_kp_v6)
            
            log_container_kp_v6.empty()
            with st.expander("카카오페이지 추출 전체 상세 처리 로그 보기", expanded=True):
                for log_line_kp_v6 in process_logs_kp_v6:
                    if "ID:" in log_line_kp_v6 and "시작" in log_line_kp_v6 : st.markdown(f"**{log_line_kp_v6}**")
                    elif "[성공]" in log_line_kp_v6: st.success(log_line_kp_v6)
                    elif "[실패]" in log_line_kp_v6 or "[오류]" in log_line_kp_v6 or "[경고]" in log_line_kp_v6 or "오류:" in log_line_kp_v6 : st.error(log_line_kp_v6)
                    elif "소요 시간" in log_line_kp_v6 or "처리 종료" in log_line_kp_v6: st.caption(log_line_kp_v6)
                    elif "ZIP 파일" in log_line_kp_v6 or "총 소요 시간" in log_line_kp_v6: st.info(log_line_kp_v6)
                    else: st.markdown(f"`{log_line_kp_v6}`")


# ------------------------------------------------------------------------------
# O. CSV의 UID로 카카오페이지 공지사항 탭 열기 (로컬 실행용)
# ------------------------------------------------------------------------------
st.subheader("13) CSV의 UID로 카카오페이지 공지사항 탭 열기 (로컬 실행 전용)") # 번호는 기존 기능 수에 맞춰 조정

st.warning("주의: 이 기능은 Streamlit 앱을 로컬 PC에서 실행할 때만 정상적으로 동작합니다. 웹 서버에 배포된 앱에서는 사용자의 브라우저에 탭을 직접 열 수 없습니다.", icon="⚠️")

# 세션 상태에서 파일 목록 가져오기
available_files_for_tab_opening = ["--- 선택하세요 ---"] + list(st.session_state.get("file_names", {}).values())

selected_csv_for_tabs = st.selectbox(
    "UID 목록이 포함된 CSV 파일 선택",
    available_files_for_tab_opening,
    key="selectbox_csv_for_tabs"
)

# time.sleep()을 위한 입력 필드
delay_between_tabs = st.number_input(
    "각 탭을 열 때 간격 (초)",
    min_value=0.1,
    max_value=5.0,
    value=0.5, # 기본값 0.5초
    step=0.1,
    key="number_input_delay_tabs"
)

if st.button("선택한 CSV의 UID로 공지사항 탭 모두 열기", key="button_open_kakao_tabs"):
    if selected_csv_for_tabs == "--- 선택하세요 ---":
        st.error("탭을 열 CSV 파일을 선택해주세요.")
    else:
        # 선택된 표시명으로 원본 DataFrame 가져오기
        original_key_for_tabs = next(
            (k for k, v in st.session_state.get("file_names", {}).items() if v == selected_csv_for_tabs), None
        )

        if original_key_for_tabs and original_key_for_tabs in st.session_state.get("csv_dataframes", {}):
            df_for_tabs = st.session_state["csv_dataframes"][original_key_for_tabs]

            if not df_for_tabs.empty and df_for_tabs.shape[1] > 0:
                # 첫 번째 열을 UID로 가정 (문자열로 변환하고 NaN 및 빈 문자열 제거)
                uids_to_open = df_for_tabs.iloc[:, 0].astype(str).dropna()
                uids_to_open = [uid for uid in uids_to_open if uid.strip()] # 공백만 있는 UID 제거

                if not uids_to_open:
                    st.warning(f"'{selected_csv_for_tabs}' 파일의 첫 번째 열에 유효한 UID가 없습니다.")
                else:
                    base_url_notice = "https://page.kakao.com/content/{}?tab_type=notice"
                    tabs_opened_count = 0
                    
                    # Chrome 브라우저를 사용하도록 시도
                    try:
                        # 특정 브라우저를 지정하려면 해당 브라우저의 실행 파일 이름을 정확히 알아야 할 수 있습니다.
                        # 'chrome', 'firefox', 'safari', 'msie', 'opera' 등이 일반적입니다.
                        # Windows: webbrowser.register('chrome', None, webbrowser.BackgroundBrowser("C://Program Files (x86)//Google//Chrome//Application//chrome.exe")) (경로 확인 필요)
                        # macOS: webbrowser.get('chrome') 또는 webbrowser.get('open -a /Applications/Google\ Chrome.app %s')
                        # Linux: webbrowser.get('google-chrome') 또는 webbrowser.get('chromium-browser')
                        # 여기서는 일반적인 'chrome'을 시도합니다. 환경에 따라 동작하지 않을 수 있습니다.
                        browser_controller = webbrowser.get(None) # 시스템 기본 브라우저 사용
                        # browser_controller = webbrowser.get('chrome') # 특정 브라우저 시도 (설치 및 환경변수 설정 필요할 수 있음)
                    except webbrowser.Error:
                        st.error("웹 브라우저를 제어할 수 없습니다. 시스템에 기본 브라우저가 설정되어 있는지 확인해주세요.")
                        st.stop() # 오류 발생 시 중단

                    st.info(f"'{selected_csv_for_tabs}' 파일에서 {len(uids_to_open)}개의 UID에 대해 탭을 엽니다...")
                    
                    with st.spinner(f"{len(uids_to_open)}개의 탭을 여는 중... (각 탭당 {delay_between_tabs}초 대기)"):
                        for uid_item in uids_to_open:
                            url_to_open = base_url_notice.format(uid_item.strip())
                            try:
                                browser_controller.open_new_tab(url_to_open)
                                tabs_opened_count += 1
                                time.sleep(delay_between_tabs)
                            except Exception as e_open_tab:
                                st.warning(f"UID '{uid_item}'의 탭을 여는 중 오류 발생: {e_open_tab}")
                                # 일부 탭 열기 실패해도 계속 진행

                    if tabs_opened_count > 0:
                        st.success(f"총 {tabs_opened_count}개의 카카오페이지 공지사항 탭이 새 탭으로 열렸습니다 (또는 열도록 시도했습니다).")
                    else:
                        st.warning("탭을 하나도 열지 못했습니다.")
            else:
                st.warning(f"'{selected_csv_for_tabs}' 파일이 비어있거나 유효한 데이터가 없습니다.")
        else:
            st.error(f"'{selected_csv_for_tabs}' 파일을 찾을 수 없습니다. 파일 목록을 확인해주세요.")

st.markdown("---") # 다음 섹션과 구분

# ------------------------------------------------------------------------------
# 앱 하단 정보 (선택 사항)
# ------------------------------------------------------------------------------
# st.markdown("---")
# st.caption("Pinsight Utility App")
# ------------------------------------------------------------------------------
# 앱 하단 정보 (선택 사항)
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("Pinsight Utility App by YourName/Organization (문의: ...)")
