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

# [CSV 로드하기] 버튼
if st.button("CSV 로드하기"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # CSV 파일 로드
                df = pd.read_csv(uploaded_file, header=None)
                st.session_state["csv_dataframes"][uploaded_file.name] = df
                st.success(f"{uploaded_file.name}: 업로드 & 로드 성공 (행 수: {len(df)})")
            except Exception as e:
                st.error(f"{uploaded_file.name}: 로드 실패 - {e}")
    else:
        st.warning("아직 업로드된 파일이 없습니다.")

# 지금까지 로드된 CSV 목록 확인
loaded_paths = list(st.session_state["csv_dataframes"].keys())

# 파일 목록과 행 수 표시
if loaded_paths:
    st.write("### 업로드된 파일 목록 (행 수 포함):")
    for file_name, df in st.session_state["csv_dataframes"].items():
        st.write(f"- {file_name} (행 수: {len(df)})")

# ------------------------------------------------------------------------------
# B. 공통 함수 - UID 집합 반환 (CSV 첫 열 기준)
# ------------------------------------------------------------------------------
def get_uid_set(csv_key):
    df = st.session_state["csv_dataframes"][csv_key]
    return set(df.iloc[:, 0].astype(str))

def save_to_session_and_download(result_df, file_name="result.csv"):
    """결과를 세션에 추가하고 다운로드 버튼 생성"""
    # 결과를 세션 상태에 추가 (자동 업로드)
    st.session_state["csv_dataframes"][file_name] = result_df

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
# C. 교집합
# ------------------------------------------------------------------------------
st.subheader("2) 교집합 (Intersection)")

selected_for_intersect = st.multiselect(
    "교집합 대상 CSV 선택 (2개 이상)",
    loaded_paths
)

if st.button("교집합 실행하기"):
    if len(selected_for_intersect) < 2:
        st.error("교집합은 2개 이상 선택해야 합니다.")
    else:
        base = get_uid_set(selected_for_intersect[0])
        for p in selected_for_intersect[1:]:
            base = base.intersection(get_uid_set(p))
        st.write(f"교집합 결과 UID 수: {len(base)}")

        # 결과 저장 및 업로드 목록에 추가
        result_df = pd.DataFrame(sorted(base))
        save_to_session_and_download(result_df, "result_intersection.csv")

# ------------------------------------------------------------------------------
# D. 조합 (합집합 + 제외)
# ------------------------------------------------------------------------------
st.subheader("3) 조합하기 (포함/제외)")

col1, col2 = st.columns(2)
with col1:
    plus_list = st.multiselect("포함(+)할 CSV들", loaded_paths)
with col2:
    minus_list = st.multiselect("제외(-)할 CSV들", loaded_paths)

if st.button("조합하기 실행"):
    if not plus_list:
        st.error("최소 1개 이상 '포함' 리스트를 지정해야 합니다.")
    else:
        plus_set = set()
        for p in plus_list:
            plus_set = plus_set.union(get_uid_set(p))

        minus_set = set()
        for m in minus_list:
            minus_set = minus_set.union(get_uid_set(m))

        final = plus_set - minus_set
        st.write(f"조합 결과 UID 수: {len(final)}")

        # 결과 저장 및 업로드 목록에 추가
        result_df = pd.DataFrame(sorted(final))
        save_to_session_and_download(result_df, "result_combination.csv")

# ------------------------------------------------------------------------------
# E. 중복 제거 (한 CSV만)
# ------------------------------------------------------------------------------
st.subheader("4) 중복 제거")

selected_for_dedup = st.selectbox("중복 제거 대상 CSV 선택 (1개만)", loaded_paths)
if st.button("중복 제거 실행"):
    df_original = st.session_state["csv_dataframes"][selected_for_dedup]
    df_dedup = df_original.drop_duplicates()
    st.write(f"원본 행 수: {len(df_original)}, 중복 제거 후: {len(df_dedup)}")

    # 결과 저장 및 업로드 목록에 추가
    save_to_session_and_download(df_dedup, "result_dedup.csv")

# ------------------------------------------------------------------------------
# F. 랜덤 추출 (샘플링)
# ------------------------------------------------------------------------------
st.subheader("5) 랜덤 추출 (샘플링)")

selected_for_sampling = st.selectbox("샘플링 대상 CSV 선택 (1개만)", loaded_paths)
sample_size = st.number_input("추출할 개수(n)", min_value=1, value=10)

if st.button("랜덤 추출 실행"):
    df_target = st.session_state["csv_dataframes"][selected_for_sampling]
    if sample_size > len(df_target):
        st.error(f"CSV 행수({len(df_target)})보다 많은 {sample_size}개를 추출할 수 없습니다.")
    else:
        df_sample = df_target.sample(n=sample_size, random_state=None)
        st.write(f"총 {len(df_target)} 중 {sample_size}개를 무작위 추출했습니다.")

        # 결과 저장 및 업로드 목록에 추가
        save_to_session_and_download(df_sample, "result_sample.csv")
