import streamlit as st
import pandas as pd
import os

# ------------------------------------------------------------------------------
# (1) 폴더 경로에 저장할 수 있도록, 다운로드 폴더 대신 사용자가 입력한 폴더에 결과 저장
# ------------------------------------------------------------------------------
# 더 이상 사용하지 않으므로 주석 처리합니다.
# DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

st.set_page_config(page_title="CSV Utility App", layout="wide")
st.title("CSV 파일 조작 앱 (교집합 / 합집합 / 중복제거 / 랜덤추출)")

# ------------------------------------------------------------------------------
# (A) CSV 파일 불러오기 (폴더 전체 / 개별 경로)
# ------------------------------------------------------------------------------
st.subheader("1) CSV 경로 또는 폴더 입력")

# 폴더 경로 입력 필드
folder_path = st.text_input("CSV 파일이 있는 폴더 경로", "")

# CSV 경로를 담을 리스트
csv_paths = []

# -----------------------------
# (A-1) 폴더 내 CSV 자동 탐색
# -----------------------------
if folder_path:
    if os.path.isdir(folder_path):
        # 폴더 내 모든 CSV 파일 탐색
        folder_csvs = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".csv")
        ]
        if folder_csvs:
            st.success(f"폴더 내 {len(folder_csvs)}개의 CSV 파일을 찾았습니다.")
            csv_paths.extend(folder_csvs)
        else:
            st.warning("폴더에 CSV 파일이 없습니다.")
    else:
        st.error("유효한 폴더 경로를 입력해주세요.")

# -----------------------------
# (A-2) 개별 CSV 경로 입력(선택사항)
#       - 굳이 제한 없이 여러 개를 받고 싶다면 아래 로직을 원하는 방식으로 수정하세요.
#       - 예: 파일 하나씩 업로드할 수 있게끔 st.file_uploader 등을 쓰는 방법도 있음
# -----------------------------
st.markdown("---")
st.write("추가로 개별 CSV 파일 경로가 있다면 아래에 입력해주세요 (선택).")
new_csv_path = st.text_input("추가 CSV 파일 경로", "")
if new_csv_path.strip():
    csv_paths.append(new_csv_path.strip())

st.write(f"총 {len(csv_paths)}개의 CSV 경로가 감지되었습니다.")

# ------------------------------------------------------------------------------
# (B) CSV 로드하기
# ------------------------------------------------------------------------------
if "csv_dataframes" not in st.session_state:
    st.session_state["csv_dataframes"] = {}  # {경로: pd.DataFrame}

def load_csv_to_session(csv_list):
    """리스트로 들어온 CSV 경로들을 읽어서 session_state에 저장"""
    for cpath in csv_list:
        if cpath not in st.session_state["csv_dataframes"]:
            try:
                df = pd.read_csv(cpath, header=None)  # CSV가 1열만 있다고 가정
                st.session_state["csv_dataframes"][cpath] = df
                st.success(f"[{cpath}] 로드 성공 (행 수: {len(df)})")
            except Exception as e:
                st.error(f"[{cpath}] 로드 실패: {e}")

# 로드 버튼
if st.button("CSV 로드하기"):
    load_csv_to_session(csv_paths)

# 지금까지 읽어온 CSV 목록 확인
loaded_paths = list(st.session_state["csv_dataframes"].keys())

if not loaded_paths:
    st.warning("아직 로드된 CSV가 없습니다. 위에서 경로 입력 후 [CSV 로드하기]를 눌러주세요.")
else:
    # ------------------------------------------------------------------------------
    # (C) 공통 함수 - UID 집합 반환
    # ------------------------------------------------------------------------------
    def get_uid_set(csv_path):
        """로딩된 CSV에서 UID 컬럼(set 형태) 반환"""
        df = st.session_state["csv_dataframes"][csv_path]
        return set(df.iloc[:, 0].astype(str))

    # ------------------------------------------------------------------------------
    # (D) 결과 저장 함수
    #     (2) 결과파일을 폴더에 저장 & 자동으로 로드
    # ------------------------------------------------------------------------------
    def save_and_notify(result_set, out_name="result.csv"):
        """결과 집합을 CSV로 저장하고, 곧바로 session_state에 로드"""
        if not folder_path:
            st.error("폴더 경로가 설정되지 않아 결과를 저장할 수 없습니다.")
            return

        result_df = pd.DataFrame(sorted(result_set))  # UID 정렬해서 저장(옵션)
        save_path = os.path.join(folder_path, out_name)  # 폴더 경로 안에 저장

        try:
            result_df.to_csv(save_path, index=False, header=False)
            st.success(f"결과 저장 완료: {save_path}")

            # (2) “바로 그 결과가 추가되면 좋겠어” → 새로 생성된 CSV를 곧바로 로드
            # 이미 session_state에 없는 경로이므로, load 함수 써서 다시 읽어들임
            load_csv_to_session([save_path])

        except Exception as e:
            st.error(f"결과 저장 실패: {e}")

    # ------------------------------------------------------------------------------
    # (E) 교집합(Intersection)
    # ------------------------------------------------------------------------------
    st.subheader("2) 교집합 (Intersection)")

    selected_for_intersect = st.multiselect(
        "교집합 대상 CSV 선택 (2개 이상)",
        loaded_paths
    )

    if st.button("교집합 실행하기"):
        if len(selected_for_intersect) < 2:
            st.error("교집합은 최소 2개 이상 CSV를 선택해야 합니다.")
        else:
            base = get_uid_set(selected_for_intersect[0])
            for p in selected_for_intersect[1:]:
                base = base.intersection(get_uid_set(p))
            st.write(f"교집합 결과 UID 수: {len(base)}")
            save_and_notify(base, "result_intersection.csv")

    # ------------------------------------------------------------------------------
    # (F) 조합하기 (합집합 + 제외)
    # ------------------------------------------------------------------------------
    st.subheader("3) 조합하기 (포함/제외)")

    st.write("아래에서 **포함(+)할 CSV**와 **제외(-)할 CSV**를 골라주세요.")
    col1, col2 = st.columns(2)

    with col1:
        plus_list = st.multiselect("포함(+)할 CSV", loaded_paths)
    with col2:
        minus_list = st.multiselect("제외(-)할 CSV", loaded_paths)

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
            save_and_notify(final, "result_combination.csv")

    # ------------------------------------------------------------------------------
    # (G) 중복제거
    # ------------------------------------------------------------------------------
    st.subheader("4) 중복 제거 (한 CSV만)")

    selected_for_dedup = st.selectbox("중복 제거 대상 CSV 선택", loaded_paths)
    if st.button("중복 제거 실행"):
        df_original = st.session_state["csv_dataframes"][selected_for_dedup]
        df_dedup = df_original.drop_duplicates()
        st.write(f"원본 행 수: {len(df_original)}, 중복 제거 후: {len(df_dedup)}")

        result_set = set(df_dedup.iloc[:, 0].astype(str))
        save_and_notify(result_set, "result_dedup.csv")

    # ------------------------------------------------------------------------------
    # (H) 랜덤 추출 (샘플링)
    # ------------------------------------------------------------------------------
    st.subheader("5) 랜덤 추출 (샘플링)")

    selected_for_sampling = st.selectbox("샘플링 대상 CSV 선택", loaded_paths)
    sample_size = st.number_input("추출할 개수(n)", min_value=1, value=10)

    if st.button("랜덤 추출 실행"):
        df_target = st.session_state["csv_dataframes"][selected_for_sampling]
        if sample_size > len(df_target):
            st.error(f"CSV 행수({len(df_target)})보다 많은 {sample_size}개를 추출할 수 없습니다.")
        else:
            df_sample = df_target.sample(n=sample_size, random_state=None)
            st.write(f"총 {len(df_target)} 중 {sample_size}개를 무작위 추출했습니다.")
            result_set = set(df_sample.iloc[:, 0].astype(str))
            save_and_notify(result_set, "result_sample.csv")