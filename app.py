import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="CSV Utility App", layout="wide")
st.title("CSV 파일 조작 앱 (교집합 / 합집합 / 중복제거 / 랜덤추출)")

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

# 세션 상태 초기화: 업로드된 CSV 파일
if "csv_dataframes" not in st.session_state:
    st.session_state["csv_dataframes"] = {}  # {filename: pd.DataFrame}
if "file_names" not in st.session_state:
    st.session_state["file_names"] = {}      # {original_filename: current_filename}

# [CSV 로드하기] 버튼
if st.button("CSV 로드하기"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # CSV 파일 로드 (header=None 주의)
                df = pd.read_csv(uploaded_file, header=None)
                st.session_state["csv_dataframes"][uploaded_file.name] = df

                # 파일명 관리
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

        # 파일명 수정 입력 필드
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

        # 행 수 표시
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
# C. 공통 함수 - UID 집합 반환 (CSV 첫 열 기준)
# ------------------------------------------------------------------------------
def get_uid_set(csv_key):
    """csv_key(오리지널 파일명)에 해당하는 df의 첫 번째 컬럼으로부터 UID set을 생성"""
    df = st.session_state["csv_dataframes"][csv_key]
    return set(df.iloc[:, 0].astype(str))

def save_to_session_and_download(result_df, file_name="result.csv"):
    """결과 df를 세션에 저장하고, 다운로드 버튼을 생성"""
    # 혹시 같은 파일명이 이미 세션에 있다면 중복을 피한다
    if file_name in st.session_state["file_names"].values():
        file_name = f"{file_name.split('.')[0]}_new.csv"
    st.session_state["csv_dataframes"][file_name] = result_df
    st.session_state["file_names"][file_name] = file_name

    # 다운로드 버튼
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False, header=False)
    st.download_button(
        label=f"{file_name} 다운로드",
        data=csv_buffer.getvalue(),
        file_name=file_name,
        mime="text/csv"
    )
    st.success(f"결과 CSV '{file_name}' 세션에 추가 완료!")

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
        # 교집합 구하기
        base = None
        for current_file_name in selected_for_intersect:
            # current_file_name에 해당하는 original key 찾기
            original_key = [k for k, v in st.session_state["file_names"].items() if v == current_file_name][0]
            if base is None:
                base = get_uid_set(original_key)
            else:
                base = base.intersection(get_uid_set(original_key))

        st.write(f"교집합 결과 UID 수: {len(base)}")
        # 결과 저장
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
        # 포함(+) 집합
        plus_set = set()
        for p in plus_list:
            original_key = [k for k, v in st.session_state["file_names"].items() if v == p][0]
            plus_set = plus_set.union(get_uid_set(original_key))

        # 제외(-) 집합
        minus_set = set()
        for m in minus_list:
            original_key = [k for k, v in st.session_state["file_names"].items() if v == m][0]
            minus_set = minus_set.union(get_uid_set(original_key))

        # 최종 결과
        final = plus_set - minus_set
        st.write(f"조합 결과 UID 수: {len(final)}")

        result_df = pd.DataFrame(sorted(final))
        save_to_session_and_download(result_df, "result_combination.csv")

# ------------------------------------------------------------------------------
# F. 중복 제거 (Unique)
# ------------------------------------------------------------------------------
st.subheader("4) 중복 제거")

# 단일 파일에서 중복 제거
unique_target = st.selectbox(
    "중복 제거할 CSV 선택",
    ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values())
)

if st.button("중복 제거 실행하기"):
    if unique_target == "--- 선택하세요 ---":
        st.error("중복 제거할 파일을 선택해주세요.")
    else:
        # original_key 찾기
        original_key = [k for k, v in st.session_state["file_names"].items() if v == unique_target][0]
        df = st.session_state["csv_dataframes"][original_key]

        # pandas drop_duplicates()로 중복 제거
        df_unique = df.drop_duplicates(keep="first")

        st.write(f"원본 행 수: {len(df)}, 중복 제거 후 행 수: {len(df_unique)}")
        save_to_session_and_download(df_unique, "result_unique.csv")

# ------------------------------------------------------------------------------
# G. 랜덤 추출
# ------------------------------------------------------------------------------
st.subheader("5) 랜덤 추출")

random_targets = st.multiselect(
    "랜덤 추출 대상 CSV 선택 (2개 이상 가능)",
    list(st.session_state["file_names"].values())
)
sample_size = st.number_input("랜덤 추출 개수", min_value=1, value=10, step=1)

if st.button("랜덤 추출 실행하기"):
    if not random_targets:
        st.error("최소 1개 이상의 파일을 선택해주세요.")
    else:
        # 여러 파일을 하나로 통합
        combined_df = pd.DataFrame()
        for rt in random_targets:
            original_key = [k for k, v in st.session_state["file_names"].items() if v == rt][0]
            combined_df = pd.concat([combined_df, st.session_state["csv_dataframes"][original_key]], ignore_index=True)

        if sample_size > len(combined_df):
            st.warning(f"랜덤 추출 개수({sample_size})가 총 행 수({len(combined_df)})보다 많습니다. 가능한 최대 개수만큼 추출합니다.")
            sample_size = len(combined_df)

        # 무작위 샘플 추출
        random_sample = combined_df.sample(n=sample_size, random_state=None)
        
        st.write(f"통합된 행 수: {len(combined_df)}, 랜덤 추출 개수: {len(random_sample)}")
        save_to_session_and_download(random_sample, "result_random.csv")
