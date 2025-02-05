import streamlit as st
import pandas as pd
import io
import time

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
# F. N번 이상 CSV에서 등장하는 UID 추출
# ------------------------------------------------------------------------------
st.subheader("4) N번 이상 등장하는 UID")

selected_for_nplus = st.multiselect(
    "대상 CSV 선택 (1개 이상)",
    list(st.session_state["file_names"].values())
)

threshold = st.number_input(
    "몇 개 이상의 CSV에서 등장한 UID를 찾을까요?",
    min_value=1,
    value=2,
    step=1
)

if st.button("N번 이상 등장 UID 추출"):
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

        valid_uids = [uid for uid, cnt in uid_count.items() if cnt >= threshold]
        st.write(f"{threshold}개 이상의 CSV에서 등장한 UID 수: {len(valid_uids)}")

        if valid_uids:
            result_df = pd.DataFrame(sorted(valid_uids))
            save_to_session_and_download(result_df, "result_nplus.csv")
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
# I. Bingo 당첨자 추출
# ------------------------------------------------------------------------------
st.subheader("7) Bingo 당첨자 추출")

def get_bingo_lines(n: int):
    lines = []
    # 가로
    for r in range(n):
        lines.append([r * n + c for c in range(n)])
    # 세로
    for c in range(n):
        lines.append([r * n + c for r in range(n)])
    # 대각선(좌상->우하)
    lines.append([i * n + i for i in range(n)])
    # 대각선(우상->좌하)
    lines.append([i * n + (n - 1 - i) for i in range(n)])
    return lines

bingo_size_option = st.selectbox("빙고판 크기 선택", ["2x2", "3x3", "4x4", "5x5"])
size_map = {"2x2": 2, "3x3": 3, "4x4": 4, "5x5": 5}
n = size_map[bingo_size_option]

max_bingo_lines = 2 * n + 2
required_bingo_count = st.number_input(
    "최소 달성해야 하는 빙고 줄 개수",
    min_value=1,
    max_value=max_bingo_lines,
    value=1,
    step=1
)
st.write(f"해당 빙고판의 최대 달성 가능 라인 수: {max_bingo_lines}")

st.write(f"**{bingo_size_option} 빙고판**")
cell_files = [None] * (n * n)

for row_idx in range(n):
    cols = st.columns(n)
    for col_idx in range(n):
        cell_idx = row_idx * n + col_idx
        with cols[col_idx]:
            selected_filename = st.selectbox(
                f"{cell_idx+1}번 칸 CSV",
                ["--- 선택 안함 ---"] + list(st.session_state["file_names"].values()),
                key=f"bingo_cell_{cell_idx}"
            )
            if selected_filename != "--- 선택 안함 ---":
                cell_files[cell_idx] = selected_filename
            
            if cell_files[cell_idx] is not None:
                st.markdown(
                    f"<div style='text-align:center;"
                    f"border:1px solid #999; border-radius:4px; padding:4px;"
                    f"margin-top:4px; background-color:#eee;'>"
                    f"선택된 파일: <b>{cell_files[cell_idx]}</b></div>",
                    unsafe_allow_html=True
                )

if st.button("당첨자 추출하기"):
    lines = get_bingo_lines(n)
    cell_uid_sets = []
    for idx, filename in enumerate(cell_files):
        if filename is not None:
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == filename][0]
            cell_uid_sets.append(get_uid_set(orig_key))
        else:
            cell_uid_sets.append(set())
    
    line_sets = []
    for line in lines:
        intersect_uid = None
        for cell_idx in line:
            if intersect_uid is None:
                intersect_uid = cell_uid_sets[cell_idx]
            else:
                intersect_uid = intersect_uid.intersection(cell_uid_sets[cell_idx])
        if intersect_uid is None:
            intersect_uid = set()
        line_sets.append(intersect_uid)
    
    user_line_count = {}
    for uid_set in line_sets:
        for uid in uid_set:
            user_line_count[uid] = user_line_count.get(uid, 0) + 1
    
    winners = [uid for uid, cnt in user_line_count.items() if cnt >= required_bingo_count]
    
    st.write(f"**당첨자 수: {len(winners)}명**")
    if winners:
        st.write("당첨자 목록 (일부만 표시):", winners[:50], "...")
        result_df = pd.DataFrame(sorted(winners))
        save_to_session_and_download(result_df, "bingo_winners.csv")
    else:
        st.warning("해당 조건을 만족하는 유저가 없습니다.")

# ------------------------------------------------------------------------------
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
# L. 코호트 분석
# ------------------------------------------------------------------------------
st.subheader("10) 코호트 분석 (Cohort Analysis)")

st.markdown(
    """
    **코호트 분석을 위한 CSV 파일 순서를 지정하세요.**  
    먼저 참여한 순서대로 파일을 선택하면,  
    - **잔존율**: 첫 번째 파일의 UID가 순차적으로 남아있는지를 계산  
    - **단순 교집합**: 첫 번째 파일을 기준으로 각 후속 파일과의 교집합 및  
      (중간 누락을 고려한) 교집합을 계산  
    합니다.
    """
)

# 코호트에 사용할 파일 개수 지정 (최소 2개)
num_cohort = st.number_input("코호트 분석에 사용할 파일 개수", min_value=2, value=2, step=1)

# 파일 순서대로 선택 (각 selectbox의 key는 고유해야 함)
cohort_files = []
for i in range(int(num_cohort)):
    file_sel = st.selectbox(
        f"{i+1}번째 참여 CSV 선택",
        ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values()),
        key=f"cohort_{i}"
    )
    if file_sel != "--- 선택하세요 ---":
        cohort_files.append(file_sel)

# 분석 방법 선택: 잔존율 vs 단순 교집합
analysis_method = st.radio(
    "코호트 분석 방법 선택",
    ("잔존율 (Retention)", "단순 교집합 (Simple Intersection)")
)

if st.button("코호트 분석 실행하기"):
    if len(cohort_files) < 2:
        st.error("최소 2개의 CSV 파일을 순서대로 선택해주세요.")
    else:
        # 첫 번째 파일의 UID set
        def get_uid_set_from_filename(filename):
            # 주어진 파일명을 통해 원본 키 찾기
            orig_key = [k for k, v in st.session_state["file_names"].items() if v == filename][0]
            df = st.session_state["csv_dataframes"][orig_key]
            # 첫 번째 컬럼을 UID로 사용 (문자열로 변환)
            return set(df.iloc[:, 0].astype(str))
        
        # 결과 DataFrame 생성
        if analysis_method == "잔존율 (Retention)":
            retention_counts = []
            retention_rate = []
            base_uids = get_uid_set_from_filename(cohort_files[0])
            initial_count = len(base_uids)
            retention_counts.append(initial_count)
            retention_rate.append(100.0)  # 100% for first cohort
            current_uids = base_uids.copy()
            # 순차적으로 교집합 계산
            for idx, filename in enumerate(cohort_files[1:], start=2):
                current_uids = current_uids.intersection(get_uid_set_from_filename(filename))
                cnt = len(current_uids)
                retention_counts.append(cnt)
                rate = (cnt / initial_count * 100) if initial_count else 0
                retention_rate.append(round(rate, 2))
            cohort_df = pd.DataFrame({
                "코호트 단계": [f"{i}차 참여" for i in range(1, len(cohort_files)+1)],
                "잔존 UID 수": retention_counts,
                "잔존율 (%)": retention_rate
            })
            st.write("### 잔존율 분석 결과")
            st.dataframe(cohort_df)
            save_to_session_and_download(cohort_df, "cohort_retention.csv")
        
        else:  # 단순 교집합 (Simple Intersection)
            base_uids = get_uid_set_from_filename(cohort_files[0])
            intersection_list = []
            # 계산: 각 후속 파일과 base 파일의 교집합,
            # 그리고 누락(중간 파일에 없는) UID를 고려한 교집합
            for idx, filename in enumerate(cohort_files[1:], start=2):
                current_uids = get_uid_set_from_filename(filename)
                # 단순 교집합: base ∩ current
                simple_intersection = base_uids.intersection(current_uids)
                
                # 누락 고려 교집합: base에서 이전(2번째 ~ (idx-1)번째) 파일들의 합집합에 해당하는 UID를 제외한 후,
                # current 파일과 교집합을 구함.
                if idx == 2:
                    adjusted_intersection = simple_intersection  # 첫 비교이므로 동일
                else:
                    # 이전에 선택된 중간 파일들의 UID union
                    previous_union = set()
                    for inter_idx in range(1, idx-1):
                        previous_union = previous_union.union(get_uid_set_from_filename(cohort_files[inter_idx]))
                    adjusted_base = base_uids - previous_union
                    adjusted_intersection = adjusted_base.intersection(current_uids)
                
                intersection_list.append({
                    "비교": f"파일1 vs 파일{idx}",
                    "단순 교집합 UID 수": len(simple_intersection),
                    "누락 고려 교집합 UID 수": len(adjusted_intersection)
                })
            
            cohort_df = pd.DataFrame(intersection_list)
            st.write("### 단순 교집합 분석 결과")
            st.dataframe(cohort_df)
            save_to_session_and_download(cohort_df, "cohort_simple_intersection.csv")
