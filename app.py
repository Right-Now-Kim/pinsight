import math
import streamlit as st
import pandas as pd
import io
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib


st.set_page_config(page_title="CSV Utility App", layout="wide")
st.title("CSV 파일 조작 앱")

# ------------------------------------------------------------------------------
# A. CSV 업로드 처리
# ------------------------------------------------------------------------------
st.subheader("1) CSV 업로드")

# 여러 파일 업로드 가능
uploaded_files = st.file_uploader(
    "여기에 CSV 파일들을 드래그하거나, 'Browse files' 버튼을 눌러 선택하세요.",
    type=["csv"],
    accept_multiple_files=True
)

# 세션 상태 초기화
if "csv_dataframes" not in st.session_state:
    st.session_state["csv_dataframes"] = {}  # {original_filename: pd.DataFrame}
if "file_names" not in st.session_state:
    st.session_state["file_names"] = {}      # {original_filename: current_filename}

# [CSV 로드하기] 버튼
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

    for original_name, df in st.session_state["csv_dataframes"].items():
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            new_name = st.text_input(
                f"파일명 변경 ({original_name})",
                value=st.session_state["file_names"].get(original_name, original_name),
                key=f"file_name_{original_name}"
            )
            # 파일명 중복 방지
            if new_name in st.session_state["file_names"].values() and new_name != original_name:
                st.warning(f"'{new_name}' 파일명이 이미 존재합니다. 다른 이름을 입력해주세요.")
            else:
                st.session_state["file_names"][original_name] = new_name

        with col2:
            st.write(f"행 수: {len(df)}")

        # 다운로드 버튼
        with col3:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, header=False)
            st.download_button(
                label="다운로드",
                data=csv_buffer.getvalue(),
                file_name=new_name,
                mime="text/csv"
            )

# ------------------------------------------------------------------------------
# C. 공통 함수
# ------------------------------------------------------------------------------
def get_uid_set(csv_key):
    """csv_key(오리지널 파일명)에 해당하는 df의 첫 번째 컬럼으로부터 UID set 생성"""
    df = st.session_state["csv_dataframes"][csv_key]
    return set(df.iloc[:, 0].astype(str))

def save_to_session_and_download(result_df, file_name="result.csv"):
    """
    결과 DataFrame을 세션에 저장하고, 다운로드 버튼을 생성합니다.
    만약 동일한 file_name이 이미 존재하면, 고유한 이름을 생성합니다.
    """
    unique_file_name = file_name
    counter = 1
    # st.session_state["csv_dataframes"]의 키(파일명)들을 확인해서 고유한 이름 만들기
    while unique_file_name in st.session_state["csv_dataframes"]:
        base, ext = unique_file_name.rsplit('.', 1)
        unique_file_name = f"{base}_{counter}.{ext}"
        counter += 1

    st.session_state["csv_dataframes"][unique_file_name] = result_df
    st.session_state["file_names"][unique_file_name] = unique_file_name

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False, header=False)
    st.download_button(
        label=f"{unique_file_name} 다운로드",
        data=csv_buffer.getvalue(),
        file_name=unique_file_name,
        mime="text/csv"
    )
    st.success(f"결과 CSV '{unique_file_name}' 세션에 추가 완료!")
# ------------------------------------------------------------------------------
# D. 교집합
# ------------------------------------------------------------------------------
st.subheader("2) 교집합 (Intersection)")

selected_for_intersect = st.multiselect(
    "교집합 대상 CSV 선택 (2개 이상)",
    list(st.session_state["file_names"].values())
)

if st.button("교집합 실행하기"):
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
        result_df = pd.DataFrame(sorted(base))
        save_to_session_and_download(result_df, "result_intersection.csv")

# ------------------------------------------------------------------------------
# E. 조합 (합집합 + 제외)
# ------------------------------------------------------------------------------
st.subheader("3) 조합하기 (포함/제외)")

col1, col2 = st.columns(2)
with col1:
    plus_list = st.multiselect("포함(+)할 CSV들", list(st.session_state["file_names"].values()))
with col2:
    minus_list = st.multiselect("제외(-)할 CSV들", list(st.session_state["file_names"].values()))

if st.button("조합하기 실행"):
    if not plus_list:
        st.error("최소 1개 이상 '포함' 리스트를 지정해야 합니다.")
    else:
        plus_set = set()
        for p in plus_list:
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == p][0]
            plus_set = plus_set.union(get_uid_set(orig_key))

        minus_set = set()
        for m in minus_list:
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == m][0]
            minus_set = minus_set.union(get_uid_set(orig_key))

        final = plus_set - minus_set
        st.write(f"조합 결과 UID 수: {len(final)}")

        result_df = pd.DataFrame(sorted(final))
        save_to_session_and_download(result_df, "result_combination.csv")

# ------------------------------------------------------------------------------
# F. N번 이상 또는 정확히 N번 CSV에서 등장하는 UID 추출
# ------------------------------------------------------------------------------
st.subheader("4) N번 이상 / 정확히 N번 등장하는 UID 추출")

selected_for_nplus = st.multiselect(
    "대상 CSV 선택 (1개 이상)",
    list(st.session_state["file_names"].values())
)

threshold = st.number_input(
    "몇 개의 CSV 파일에서 등장하는 UID를 찾을까요?",
    min_value=1,
    value=2,
    step=1
)

# 조건 선택: "N번 이상 등장" vs "정확히 N번 등장"
condition_option = st.radio(
    "추출 조건 선택",
    ("N번 이상 등장", "정확히 N번 등장")
)

if st.button("UID 추출 실행"):
    if not selected_for_nplus:
        st.error("최소 1개 이상의 CSV 파일을 선택해주세요.")
    else:
        if threshold > len(selected_for_nplus):
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
        else:  # "정확히 N번 등장"
            valid_uids = [uid for uid, cnt in uid_count.items() if cnt == threshold]
            st.write(f"정확히 {threshold}개 등장하는 UID 수: {len(valid_uids)}")

        if valid_uids:
            result_df = pd.DataFrame(sorted(valid_uids))
            filename = "result_nplus.csv" if condition_option == "N번 이상 등장" else "result_exactly_n.csv"
            save_to_session_and_download(result_df, filename)
        else:
            st.warning("해당 조건을 만족하는 UID가 없습니다.")

# ------------------------------------------------------------------------------
# G. 중복 제거 (Unique)
# ------------------------------------------------------------------------------
st.subheader("5) 중복 제거")

unique_target = st.selectbox(
    "중복 제거할 CSV 선택",
    ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values())
)

if st.button("중복 제거 실행하기"):
    if unique_target == "--- 선택하세요 ---":
        st.error("중복 제거할 파일을 선택해주세요.")
    else:
        orig_key = [k for k, v in st.session_state["file_names"].items() if v == unique_target][0]
        df = st.session_state["csv_dataframes"][orig_key]

        df_unique = df.drop_duplicates(keep="first")

        st.write(f"원본 행 수: {len(df)}, 중복 제거 후 행 수: {len(df_unique)}")
        save_to_session_and_download(df_unique, "result_unique.csv")

# ------------------------------------------------------------------------------
# H. 랜덤 추출
# ------------------------------------------------------------------------------
st.subheader("6) 랜덤 추출")

random_targets = st.multiselect(
    "랜덤 추출 대상 CSV 선택 (1개 이상)",
    list(st.session_state["file_names"].values())
)
sample_size = st.number_input("랜덤 추출 개수", min_value=1, value=10, step=1)

if st.button("랜덤 추출 실행하기"):
    if not random_targets:
        st.error("최소 1개 이상의 파일을 선택해주세요.")
    else:
        combined_df = pd.DataFrame()
        for rt in random_targets:
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == rt][0]
            combined_df = pd.concat([combined_df, st.session_state["csv_dataframes"][orig_key]], ignore_index=True)

        if sample_size > len(combined_df):
            st.warning(f"랜덤 추출 개수({sample_size})가 총 행 수({len(combined_df)})보다 많습니다. 가능한 최대 개수만큼 추출합니다.")
            sample_size = len(combined_df)

        random_sample = combined_df.sample(n=sample_size, random_state=None)
        
        st.write(f"통합된 행 수: {len(combined_df)}, 랜덤 추출 개수: {len(random_sample)}")
        save_to_session_and_download(random_sample, "result_random.csv")
        
# ------------------------------------------------------------------------------
# I. Bingo 당첨자 추출 (시각화 및 파일 저장 기능 강화)
# ------------------------------------------------------------------------------
st.subheader("7) Bingo 당첨자 추출")

# 올바른 인덱스 기준으로 빙고 줄 생성
def get_bingo_lines(n: int):
    lines = []
    # 가로 줄
    for r in range(n):
        lines.append([r * n + c for c in range(n)])
    # 세로 줄
    for c in range(n):
        lines.append([r * n + c for r in range(n)])
    # 대각선 (좌상->우하)
    lines.append([i * n + i for i in range(n)])
    # 대각선 (우상->좌하)
    lines.append([i * n + (n - 1 - i) for i in range(n)])
    return lines

# UI: 셀별 파일 선택 결과를 순서대로 저장
# (별도 코드에서 cell_files[cell_idx] = selected_filename)

if st.button("당첨자 추출하기"):
    # 1) 각 셀별 UID 집합 및 개수 계산
    cell_uid_sets = []
    cell_uid_counts = []
    for filename in cell_files:
        if filename:
            orig_key = next(k for k,v in st.session_state["file_names"].items() if v == filename)
            uid_set = set(st.session_state["csv_dataframes"][orig_key].iloc[:,0].astype(str))
            cell_uid_sets.append(uid_set)
            cell_uid_counts.append(len(uid_set))
        else:
            cell_uid_sets.append(set())
            cell_uid_counts.append(0)

    # 2) 빙고 라인별 교집합 계산
    lines = get_bingo_lines(n)
    line_sets = []
    for line in lines:
        # 첫 셀을 복사한 뒤 나머지와 교집합
        intersect_uid = cell_uid_sets[line[0]].copy()
        for idx in line[1:]:
            intersect_uid &= cell_uid_sets[idx]
        line_sets.append(intersect_uid)

    # 3) 각 UID별 달성한 빙고 줄 수 집계
    user_line_count = {}
    for uid_set in line_sets:
        for uid in uid_set:
            user_line_count[uid] = user_line_count.get(uid, 0) + 1

    # 4) 시각화: 빙고판 및 달성 라인 그리기
    fig, ax = plt.subplots(figsize=(n*2, n*2))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n+1))
    ax.set_yticks(range(n+1))
    ax.grid(True)

    # 셀 그리기
    for idx in range(n*n):
        row, col = divmod(idx, n)
        rect = Rectangle((col, n-row-1), 1, 1, fill=False, edgecolor='gray', linewidth=1)
        ax.add_patch(rect)
        text = f"{cell_files[idx]}\nUID: {cell_uid_counts[idx]}" if cell_files[idx] else "---"
        ax.text(col+0.5, n-row-0.5, text, ha='center', va='center', fontsize=10)

    # 빙고 라인 그리기 및 달성자 수 표시
    colors = ['red','blue','green','orange','purple','brown','cyan','magenta','lime','pink','teal','olive']
    for i, line in enumerate(lines):
        pts_x = [(idx % n) + 0.5 for idx in line]
        pts_y = [n - (idx // n) - 0.5 for idx in line]
        color = colors[i % len(colors)]
        ax.plot(pts_x, pts_y, color=color, linewidth=3)
        count = len(line_sets[i])
        if count:
            mid_x = sum(pts_x)/len(pts_x)
            mid_y = sum(pts_y)/len(pts_y)
            ax.text(mid_x, mid_y, str(count), color=color, fontsize=12, fontweight='bold',
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor=color))

    ax.set_title("빙고판 시각화")
    ax.axis('off')
    st.pyplot(fig)

    # 5) 결과 출력 (데이터프레임 등)
    bingo_info = []
    for idx, line in enumerate(lines):
        bingo_info.append({
            "빙고 번호": idx+1,
            "해당 칸 인덱스": line,
            "달성자 수": len(line_sets[idx])
        })
    bingo_info_df = pd.DataFrame(bingo_info)
    st.dataframe(bingo_info_df)

    # (이후 원하는 추출 방식에 따라 user_line_count 활용)

# J. 특정 열(컬럼) 삭제하기
# ------------------------------------------------------------------------------
st.subheader("8) 특정 열(컬럼) 삭제하기")

column_delete_target = st.selectbox(
    "열 삭제할 CSV를 선택하세요",
    ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values())
)

use_header = st.checkbox("CSV 첫 행을 헤더로 사용하기 (체크 시, 첫 행을 컬럼명으로 간주)")

if column_delete_target != "--- 선택하세요 ---":
    orig_key = [k for k, v in st.session_state["file_names"].items() if v == column_delete_target][0]
    df = st.session_state["csv_dataframes"][orig_key]
    df_temp = df.copy()

    if use_header and len(df_temp) > 0:
        df_temp.columns = df_temp.iloc[0]
        df_temp = df_temp[1:].reset_index(drop=True)

    columns_list = df_temp.columns.tolist()
    selected_cols_to_delete = st.multiselect(
        "삭제할 열을 선택하세요",
        options=columns_list
    )

    if st.button("열 삭제 실행"):
        if not selected_cols_to_delete:
            st.warning("최소 한 개 이상의 열을 선택해 주세요.")
        else:
            df_temp.drop(columns=selected_cols_to_delete, inplace=True, errors="ignore")
            new_file_name = f"{column_delete_target}_cols_removed.csv"
            save_to_session_and_download(df_temp, new_file_name)

# ------------------------------------------------------------------------------
# K. N개로 분할하기
# ------------------------------------------------------------------------------
st.subheader("9) 파일 분할하기")

split_target = st.selectbox(
    "분할할 CSV를 선택하세요",
    ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values())
)
n_parts = st.number_input("몇 개로 분할할까요?", min_value=2, value=2, step=1)

if st.button("파일 분할 실행"):
    if split_target == "--- 선택하세요 ---":
        st.error("분할할 파일을 선택해 주세요.")
    else:
        orig_key = [k for k, v in st.session_state["file_names"].items() if v == split_target][0]
        df = st.session_state["csv_dataframes"][orig_key]
        total_rows = len(df)

        if n_parts > total_rows:
            st.warning(f"분할 개수({n_parts})가 전체 행 수({total_rows})보다 많습니다.\
 최대 {total_rows}개까지 분할 가능합니다.")
            n_parts = total_rows

        # 분할 로직
        base_chunk = total_rows // n_parts
        remainder = total_rows % n_parts

        st.write(f"총 행 수: {total_rows}, 분할 개수: {n_parts}, 각 파트 기본 크기: {base_chunk}, 나머지: {remainder}")

        start_idx = 0
        for i in range(n_parts):
            # i < remainder인 파트는 base_chunk+1, 나머지는 base_chunk
            chunk_size = base_chunk + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size

            # 슬라이싱
            df_chunk = df.iloc[start_idx:end_idx].reset_index(drop=True)
            part_file_name = f"{split_target}_part{i+1}.csv"
            
            # 세션에 저장 & 다운로드
            save_to_session_and_download(df_chunk, part_file_name)

            st.info(f"{part_file_name} : 행 {start_idx} ~ {end_idx-1} (총 {chunk_size}행)")

            start_idx = end_idx

# ------------------------------------------------------------------------------
# L. 사용자 지정 파일 순서 및 매트릭스 출력 (실행 버튼 및 출력 형식 선택 추가)
# ------------------------------------------------------------------------------
st.subheader("10) 사용자 지정 파일 순서 및 매트릭스 출력")

all_files = list(st.session_state["file_names"].values())
if not all_files:
    st.warning("업로드된 파일이 없습니다.")
else:
    st.markdown("**매트릭스에 사용할 파일 순서를 직접 지정해주세요.**")
    ordered_file_count = st.number_input("매트릭스에 사용할 파일 개수", 
                                           min_value=1, max_value=len(all_files), 
                                           value=2, step=1)
    
    ordered_files = []
    for i in range(int(ordered_file_count)):
        file_sel = st.selectbox(f"{i+1}번째 파일 선택", all_files, key=f"order_{i}")
        ordered_files.append(file_sel)
    
    # 중복 선택 여부 확인
    if len(set(ordered_files)) < len(ordered_files):
        st.error("같은 파일이 여러 번 선택되었습니다. 각 순서에는 서로 다른 파일을 선택해주세요.")
    else:
        # 1) 출력 형식 선택
        st.markdown("### 출력 형식 선택")
        representation_option = st.radio(
            "어떻게 결과를 표시할까요?",
            ("절대값", "절대값 (비율)", "비율"),
            key="representation_option"
        )
        
        # 2) 실행하기 버튼을 눌러야 매트릭스가 생성됨
        if st.button("실행하기"):
            # 미리 각 파일의 UID set을 계산 (첫 번째 컬럼 기준)
            uid_sets = {}
            for fname in ordered_files:
                orig_key = [k for k, v in st.session_state["file_names"].items() if v == fname][0]
                uid_sets[fname] = get_uid_set(orig_key)
            
            # ------------------------------
            # (A) 파일 간 교집합 매트릭스
            # ------------------------------
            matrix = []
            header = [""] + ordered_files
            matrix.append(header)
            
            for i, file_i in enumerate(ordered_files):
                row = [file_i]
                base_count = len(uid_sets[file_i])
                for j, file_j in enumerate(ordered_files):
                    if j < i:
                        row.append("")  # 중복 영역은 빈 칸 처리
                    else:
                        inter_count = len(uid_sets[file_i].intersection(uid_sets[file_j]))
                        # representation_option에 따라 출력 형식 결정
                        if representation_option == "절대값":
                            cell_val = inter_count
                        elif representation_option == "비율":
                            cell_val = f"{(round(inter_count/base_count*100, 2) if base_count > 0 else 0)}%"
                        else:  # "절대값 (비율)"
                            cell_val = f"{inter_count} ({round(inter_count/base_count*100, 2) if base_count > 0 else 0}%)"
                        row.append(cell_val)
                matrix.append(row)
            
            matrix_df = pd.DataFrame(matrix[1:], columns=matrix[0])
            st.write("### 파일 간 교집합 매트릭스")
            st.dataframe(matrix_df)
            save_to_session_and_download(matrix_df, "pairwise_intersection_matrix.csv")
            
            # ------------------------------
            # (B) 잔존율 매트릭스
            # ------------------------------
            retention_matrix = []
            header = [""] + ordered_files
            retention_matrix.append(header)
            
            for i, file_i in enumerate(ordered_files):
                row = [file_i]
                base_count = len(uid_sets[file_i])
                for j, file_j in enumerate(ordered_files):
                    if j < i:
                        row.append("")  # 위쪽 영역 빈 칸 처리
                    else:
                        # file_i부터 file_j까지 순차적 교집합 계산
                        current_intersection = uid_sets[file_i]
                        for k in range(i+1, j+1):
                            current_intersection = current_intersection.intersection(uid_sets[ordered_files[k]])
                        abs_val = len(current_intersection)
                        if base_count > 0:
                            rate = round(abs_val / base_count * 100, 2)
                        else:
                            rate = 0.0
                        
                        if representation_option == "절대값":
                            cell_val = abs_val
                        elif representation_option == "비율":
                            cell_val = f"{rate}%"
                        else:  # "절대값 (비율)"
                            cell_val = f"{abs_val} ({rate}%)"
                        row.append(cell_val)
                retention_matrix.append(row)
            
            retention_df = pd.DataFrame(retention_matrix[1:], columns=retention_matrix[0])
            st.write("### 잔존율 매트릭스")
            st.dataframe(retention_df)
            save_to_session_and_download(retention_df, "retention_matrix.csv")

# ------------------------------------------------------------------------------
# M. 벤 다이어그램 생성
# ------------------------------------------------------------------------------
st.subheader("M) 벤 다이어그램 생성")

selected_for_venn = st.multiselect(
    "벤 다이어그램에 사용할 CSV 파일 선택 (최대 3개)",
    list(st.session_state["file_names"].values())
)

if st.button("벤 다이어그램 생성하기"):
    if not selected_for_venn:
        st.error("최소 1개의 CSV 파일을 선택해주세요.")
    elif len(selected_for_venn) > 3:
        st.error("벤 다이어그램은 최대 3개의 집합만 지원합니다.")
    else:
        # 선택한 각 CSV 파일에 대해 UID 집합 계산 (첫 번째 컬럼 기준)
        venn_sets = {}
        for fname in selected_for_venn:
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == fname][0]
            venn_sets[fname] = get_uid_set(orig_key)
        
        import matplotlib.pyplot as plt
        from matplotlib_venn import venn2, venn3
        import matplotlib.patches as mpatches

        plt.figure(figsize=(6,6))
        
        if len(selected_for_venn) == 1:
            # 1개 집합인 경우: 단순 원으로 표시
            fname = selected_for_venn[0]
            count = len(venn_sets[fname])
            circle = plt.Circle((0.5, 0.5), 0.3, color="skyblue", alpha=0.5)
            plt.gca().add_artist(circle)
            plt.text(0.5, 0.5, f"{fname}\n{count}", horizontalalignment='center',
                     verticalalignment='center', fontsize=14, fontweight='bold')
            plt.title("벤 다이어그램 (1 집합)")
            # 범례 생성 (원형 패치)
            legend_patch = mpatches.Patch(color='skyblue', label=f"{fname}")
            plt.legend(handles=[legend_patch], loc="lower right")
        elif len(selected_for_venn) == 2:
            set1 = venn_sets[selected_for_venn[0]]
            set2 = venn_sets[selected_for_venn[1]]
            v = venn2([set1, set2], set_labels=(selected_for_venn[0], selected_for_venn[1]))
            plt.title("벤 다이어그램 (2 집합)")
            # 범례: 각 집합의 이름과 해당 원의 색상을 표시
            patch1 = mpatches.Patch(color=v.get_patch_by_id('10').get_facecolor(), label=selected_for_venn[0])
            patch2 = mpatches.Patch(color=v.get_patch_by_id('01').get_facecolor(), label=selected_for_venn[1])
            plt.legend(handles=[patch1, patch2], loc="lower right")
        else:
            # 3개 집합인 경우
            set1 = venn_sets[selected_for_venn[0]]
            set2 = venn_sets[selected_for_venn[1]]
            set3 = venn_sets[selected_for_venn[2]]
            v = venn3([set1, set2, set3], set_labels=(selected_for_venn[0], selected_for_venn[1], selected_for_venn[2]))
            plt.title("벤 다이어그램 (3 집합)")
            # 범례: 각 집합의 이름과 색상을 표시 (venn3의 영역 색상을 이용)
            # 영역 '100', '010', '001'은 각각 개별 집합의 색상
            patch1 = mpatches.Patch(color=v.get_patch_by_id('100').get_facecolor(), label=selected_for_venn[0])
            patch2 = mpatches.Patch(color=v.get_patch_by_id('010').get_facecolor(), label=selected_for_venn[1])
            patch3 = mpatches.Patch(color=v.get_patch_by_id('001').get_facecolor(), label=selected_for_venn[2])
            plt.legend(handles=[patch1, patch2, patch3], loc="lower right")
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
