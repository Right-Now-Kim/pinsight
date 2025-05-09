import math
import streamlit as st
import pandas as pd
import io
import zipfile
import time # time 모듈은 현재 코드에서 직접 사용되지 않지만, 혹시 몰라 남겨둡니다.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# import matplotlib # matplotlib.pyplot을 plt로 임포트했으므로 중복될 수 있습니다.


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
            # 현재 파일명을 제외한 다른 파일명들과 비교
            other_file_names = [v for k, v in st.session_state["file_names"].items() if k != original_name]
            if new_name in other_file_names:
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
                file_name=st.session_state["file_names"].get(original_name, original_name), # 변경된 파일명으로 다운로드
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
    # st.session_state["file_names"]의 값들 (현재 사용 중인 파일명)을 확인
    current_filenames_in_use = list(st.session_state["file_names"].values())
    
    # 새 파일명이 기존 파일명과 충돌하는지 확인 (오리지널 키가 아닌 현재 파일명 기준)
    while unique_file_name in current_filenames_in_use:
        base, ext = unique_file_name.rsplit('.', 1) if '.' in unique_file_name else (unique_file_name, '')
        if ext:
            unique_file_name = f"{base}_{counter}.{ext}"
        else:
            unique_file_name = f"{base}_{counter}"
        counter += 1

    # 새 파일명을 오리지널 키로 사용 (이 방식은 오리지널 파일명이 실제 파일의 고유 식별자가 아닐 경우 문제 소지 있음)
    # 더 나은 방법은 고유 ID를 생성하거나, unique_file_name 자체를 오리지널 키로 사용하는 것.
    # 여기서는 unique_file_name을 original_key로 사용하고, file_names 딕셔너리에도 동일하게 저장.
    # 이 경우, 오리지널 키 = 현재 파일명이 됨.
    original_key_for_new_file = unique_file_name 
    st.session_state["csv_dataframes"][original_key_for_new_file] = result_df
    st.session_state["file_names"][original_key_for_new_file] = unique_file_name


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
        result_df = pd.DataFrame(sorted(list(base))) # set을 list로 변환 후 정렬
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

        result_df = pd.DataFrame(sorted(list(final))) # set을 list로 변환 후 정렬
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
        if threshold > len(selected_for_nplus) and len(selected_for_nplus)>0 : # 추가: selected_for_nplus가 비어있지 않을 때만 경고
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

        if len(combined_df) == 0: # 추가: combined_df가 비어있는 경우 처리
            st.warning("선택한 파일들의 데이터가 비어있어 랜덤 추출을 할 수 없습니다.")
        elif sample_size > len(combined_df):
            st.warning(f"랜덤 추출 개수({sample_size})가 총 행 수({len(combined_df)})보다 많습니다. 가능한 최대 개수만큼 추출합니다.")
            sample_size = len(combined_df)
            random_sample = combined_df.sample(n=sample_size, random_state=None)
            st.write(f"통합된 행 수: {len(combined_df)}, 랜덤 추출 개수: {len(random_sample)}")
            save_to_session_and_download(random_sample, "result_random.csv")
        else:
            random_sample = combined_df.sample(n=sample_size, random_state=None)
            st.write(f"통합된 행 수: {len(combined_df)}, 랜덤 추출 개수: {len(random_sample)}")
            save_to_session_and_download(random_sample, "result_random.csv")
        
# ------------------------------------------------------------------------------
# I. Bingo 당첨자 추출 (시각화 및 파일 저장 기능 강화)
# ------------------------------------------------------------------------------
st.title("빙고 당첨자 추출")

# ─── 빙고판 크기 선택 ────────────────────────────────────────────────────────
bingo_size_options = ["2x2", "3x3", "4x4", "5x5"]
default_bingo_size_idx = 1  # 3x3
if "bingo_size_selection" in st.session_state:
    try:
        default_bingo_size_idx = bingo_size_options.index(st.session_state.bingo_size_selection)
    except ValueError:
        st.session_state.bingo_size_selection = bingo_size_options[default_bingo_size_idx]

bingo_size_selection = st.selectbox(
    "빙고판 크기 선택",
    bingo_size_options,
    index=default_bingo_size_idx,
    key="bingo_size_selector_key"
)
if st.session_state.get("bingo_size_selection") != bingo_size_selection:
    st.session_state.bingo_size_selection = bingo_size_selection
    # st.experimental_rerun() # 필요시 활성화

size_map = {"2x2": 2, "3x3": 3, "4x4": 4, "5x5": 5}
n = size_map[bingo_size_selection]

# ─── 셀별 파일 매핑 ─────────────────────────────────────────────────────────
cell_files = [None] * (n * n)
st.markdown("### 빙고판 구성 (번호 순서대로 CSV 선택)")

# st.session_state.file_names가 실제 파일명을 담고 있다고 가정 (key: 원본 파일명, value: 사용자 지정 표시명 또는 동일 파일명)
# 여기서는 value를 사용자가 선택할 수 있도록 함.
# get_available_files 함수 등이 있어서 st.session_state.get("file_names", {}) 형태로 가져온다고 가정.
# 현재 코드는 st.session_state.get("file_names", {}).values()를 사용하므로,
# file_names는 { 'original_name1.csv': 'display_name1', ... } 형태여야 함.
available_files_for_bingo = ["---"] + list(st.session_state.get("file_names", {}).values())

if len(available_files_for_bingo) > 1:
    for r_bingo in range(n):
        cols_bingo = st.columns(n)
        for c_bingo in range(n):
            idx_bingo = r_bingo * n + c_bingo
            with cols_bingo[c_bingo]:
                cell_session_key = f"bingo_cell_selection_{n}_{idx_bingo}" # 빙고 크기 변경시 키 초기화를 위해 n 추가
                selectbox_key = f"selectbox_cell_ui_{n}_{idx_bingo}"

                current_selection_for_cell = st.session_state.get(cell_session_key, "---")
                current_selection_idx = 0
                if current_selection_for_cell in available_files_for_bingo:
                    current_selection_idx = available_files_for_bingo.index(current_selection_for_cell)
                else:
                    st.session_state[cell_session_key] = "---"

                option = st.selectbox(
                    f"{idx_bingo+1}번 칸",
                    available_files_for_bingo,
                    index=current_selection_idx,
                    key=selectbox_key
                )

                if st.session_state.get(cell_session_key) != option:
                    st.session_state[cell_session_key] = option
                    # st.experimental_rerun() # 필요시 활성화

                if option != "---":
                    cell_files[idx_bingo] = option # 여기서 cell_files는 실제 표시된 파일명(display_name)
else:
    st.warning("빙고판을 구성하려면 먼저 CSV 파일을 업로드하고 로드해주세요.")


# ─── 빙고 라인 생성 함수 ───────────────────────────────────────────────────
def get_bingo_lines(n_val):
    lines = []
    for r_val in range(n_val): lines.append([r_val * n_val + c_val for c_val in range(n_val)])
    for c_val in range(n_val): lines.append([r_val * n_val + c_val for r_val in range(n_val)])
    lines.append([i * n_val + i for i in range(n_val)])
    lines.append([i * n_val + (n_val - 1 - i) for i in range(n_val)])
    return lines

# ─── 당첨자 추출 및 시각화 ───────────────────────────────────────────────────
if st.button("당첨자 추출하기", key="run_bingo_extraction_key"):
    # UI에서 최신 cell_files 값 다시 가져오기
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
    
    # cell_files 대신 current_cell_files 사용
    if not any(f for f in current_cell_files if f is not None):
        st.error("빙고판에 CSV 파일을 하나 이상 선택해주세요.")
    else:
        uid_sets, counts = [], []
        file_names_map = st.session_state.get("file_names", {}) # { 'original_name.csv': 'display_name', ... }
        csv_dataframes = st.session_state.get("csv_dataframes", {}) # { 'original_name.csv': DataFrame, ... }

        for display_name_in_cell in current_cell_files:
            if display_name_in_cell:
                # display_name으로부터 original_name 찾기
                original_file_key = next((k for k, v in file_names_map.items() if v == display_name_in_cell), None)

                if original_file_key and original_file_key in csv_dataframes:
                    df_bingo = csv_dataframes[original_file_key]
                    if not df_bingo.empty and df_bingo.shape[1] > 0:
                        s = set(df_bingo.iloc[:, 0].astype(str))
                        uid_sets.append(s)
                        counts.append(len(s))
                    else:
                        uid_sets.append(set())
                        counts.append(0)
                else:
                    uid_sets.append(set())
                    counts.append(0)
                    st.warning(f"경고: 빙고 셀 파일 '{display_name_in_cell}'에 대한 데이터를 찾을 수 없습니다 (원본 CSV 확인 필요).")
            else:
                uid_sets.append(set())
                counts.append(0)

        lines = get_bingo_lines(n)
        line_sets = []
        for ln_indices in lines:
            inter = set()
            if ln_indices and uid_sets[ln_indices[0]]: # 첫 번째 셀에 데이터가 있어야 교집합 시작 가능
                inter = uid_sets[ln_indices[0]].copy()
                for i_idx in ln_indices[1:]:
                    # 해당 셀에 데이터가 없거나(uid_sets[i_idx]가 비었음) 파일 자체가 선택 안된 경우(이 경우도 uid_sets[i_idx]는 비어있음)
                    if uid_sets[i_idx]:
                        inter &= uid_sets[i_idx]
                    else: # 중간에 빈 셀이 있으면 해당 라인은 달성자 없음
                        inter = set()
                        break
            line_sets.append(inter)

        user_count = defaultdict(int) # 각 사용자별 달성한 빙고 라인 수
        for s_set in line_sets:
            for uid_val in s_set:
                user_count[uid_val] += 1
        
        # --- 시각화 로직 (제공된 코드와 거의 동일하게 유지) ---
        fig, ax = plt.subplots(figsize=(n * 2, n * 2))
        ax.set_xlim(0, n); ax.set_ylim(0, n)
        ax.set_xticks(range(n + 1)); ax.set_yticks(range(n + 1))
        ax.grid(True, zorder=0)

        cmap = plt.get_cmap('tab20', len(lines)) # plt.cm.get_cmap 대신 plt.get_cmap 사용
        colors = [cmap(i) for i in range(len(lines))]

        for i, ln_indices in enumerate(lines):
            xs = [(idx % n) + 0.5 for idx in ln_indices]
            ys = [n - (idx // n) - 0.5 for idx in ln_indices]
            ax.plot(xs, ys, color=colors[i], linewidth=3, zorder=1, alpha=0.7)

        for idx_patch in range(n * n):
            r_patch, c_patch = divmod(idx_patch, n)
            ax.add_patch(Rectangle((c_patch, n - r_patch - 1), 1, 1, fill=False, edgecolor='gray', linewidth=1, zorder=2))
            file_display_name = current_cell_files[idx_patch] or '---' # current_cell_files 사용
            
            if len(file_display_name) > 12 and n <= 3:
                 file_display_name = file_display_name[:10] + "\n" + file_display_name[10:20]
            elif len(file_display_name) > 18 :
                 file_display_name = file_display_name[:15] + "\n" + file_display_name[15:30]

            txt = f"{file_display_name}\n({counts[idx_patch]})"
            ax.text(c_patch + 0.5, n - r_patch - 0.5, txt, ha='center', va='center', fontsize=6 if n>=4 else 8,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.9), zorder=3)

        base_offset = 0.55 if n <=3 else 0.65
        for i, ln_indices in enumerate(lines):
            if line_sets[i]:
                xs = [(idx % n) + 0.5 for idx in ln_indices]
                ys = [n - (idx // n) - 0.5 for idx in ln_indices]
                if not xs or not ys: continue

                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                angle_rad = math.atan2(my - n / 2, mx - n / 2) if n > 0 else 0
                current_offset = base_offset
                
                is_diag1 = (ln_indices == [k_val * n + k_val for k_val in range(n)])
                is_diag2 = (ln_indices == [k_val * n + (n - 1 - k_val) for k_val in range(n)])

                if (is_diag1 or is_diag2) and n > 2:
                    current_offset = base_offset * 1.3

                dx = current_offset * math.cos(angle_rad)
                dy = current_offset * math.sin(angle_rad)
                label_x, label_y = mx + dx, my + dy

                ax.text(label_x, label_y, str(len(line_sets[i])), color=colors[i], fontsize=8 if n >=4 else 10,
                        fontweight='bold', ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor=colors[i], alpha=1.0, pad=0.3), zorder=5)

        ax.set_title(f"{n}x{n} 빙고판 시각화", fontsize=12); ax.axis('off')
        st.pyplot(fig)
        # --- 시각화 로직 끝 ---

        line_names = []
        for r_idx in range(n): line_names.append(f"가로 {r_idx+1}")
        for c_idx in range(n): line_names.append(f"세로 {c_idx+1}")
        line_names.append("대각선 \\")
        line_names.append("대각선 /")

        info_data = []
        for idx, ln_indices_info in enumerate(lines):
            current_line_name = line_names[idx] if idx < len(line_names) else f"라인 {idx+1}"
            info_data.append({
                "빙고 라인": current_line_name,
                "셀 번호 (0-indexed)": str(ln_indices_info),
                "달성자 수": len(line_sets[idx])
            })
        st.dataframe(pd.DataFrame(info_data))
        st.markdown("---")

        # ----------------------------------------------------------------------
        # 1. 각 라인별 당첨자 ZIP 다운로드 (기존 기능)
        # ----------------------------------------------------------------------
        zip_lines_buf = io.BytesIO()
        lines_zip_created = False
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
                label="각 라인별 당첨자 목록 ZIP 다운로드",
                data=zip_lines_buf,
                file_name=f"bingo_{n}x{n}_lines_winners.zip",
                mime="application/zip",
                key="download_bingo_lines_zip_key"
            )
        else:
            st.info("빙고를 달성한 라인이 없어 '각 라인별 당첨자 목록'을 다운로드할 파일이 없습니다.")
        st.markdown("---")

        # ----------------------------------------------------------------------
        # 2. N 빙고 달성자 목록 ZIP 다운로드 (새로운 기능)
        #    (예: 3x3 빙고판 경우, 1빙고, 2빙고, ..., 8빙고 달성자 각각 파일로)
        # ----------------------------------------------------------------------
        st.markdown("### N-빙고 달성자 목록 (정확히 N개 라인 달성)")
        if user_count: # user_count가 비어있지 않은 경우 (빙고 달성자가 한 명이라도 있는 경우)
            winners_by_exact_bingo_count = defaultdict(list)
            for uid, count_val in user_count.items(): # count는 이미 사용된 변수명이므로 count_val 사용
                if count_val > 0:
                    winners_by_exact_bingo_count[count_val].append(uid)

            if winners_by_exact_bingo_count:
                zip_exact_buf = io.BytesIO()
                exact_zip_created = False
                with zipfile.ZipFile(zip_exact_buf, mode='w') as zf_exact:
                    max_possible_bingos = len(lines)
                    for num_bingos in range(1, max_possible_bingos + 1):
                        if num_bingos in winners_by_exact_bingo_count:
                            uids_for_exact_count = sorted(list(winners_by_exact_bingo_count[num_bingos])) # UID 정렬
                            if uids_for_exact_count:
                                df_exact_winners = pd.DataFrame(uids_for_exact_count, columns=['UID'])
                                csv_content = df_exact_winners.to_csv(index=False, header=True).encode('utf-8-sig')
                                zf_exact.writestr(f"bingo_winners_{num_bingos}_lines.csv", csv_content)
                                exact_zip_created = True
                
                if exact_zip_created:
                    zip_exact_buf.seek(0)
                    st.download_button(
                        label=f"N-빙고 달성자 목록 ZIP 다운로드 (빙고 수별 파일)",
                        data=zip_exact_buf,
                        file_name=f"bingo_{n}x{n}_winners_by_exact_line_count.zip",
                        mime="application/zip",
                        key="download_exact_bingo_count_zip_key"
                    )
                else: # 이 경우는 거의 발생하지 않음 (winners_by_exact_bingo_count가 있는데 파일이 안 만들어지는 경우)
                    st.info("N-빙고 달성자 데이터는 있으나, 다운로드할 파일을 생성하지 못했습니다.")
            else:
                st.info("빙고를 1개 이상 달성한 사용자가 없어 N-빙고별 목록을 생성할 수 없습니다.")
        else: # user_count가 비어있는 경우 (아무도 빙고를 달성하지 못함)
            st.info("빙고를 달성한 사용자가 없어 N-빙고별 목록을 생성할 수 없습니다.")
        st.markdown("---")

        # ----------------------------------------------------------------------
        # 3. N개 이상 빙고 달성자 목록 생성 및 다운로드 (기존 기능 개선)
        # ----------------------------------------------------------------------
        st.markdown("### N개 이상 빙고 달성자 목록")
        
        # number_input의 key는 고유해야 하며, 값 유지를 위해 세션 상태 사용
        num_input_key_at_least_n = f"min_bingo_lines_input_{n}" # 빙고 크기별로 키 분리
        
        # 당첨자 추출 시 number_input 값 초기화 (선택 사항)
        # if "run_bingo_extraction_key_triggered" not in st.session_state or st.session_state.run_bingo_extraction_key_triggered:
        # st.session_state[num_input_key_at_least_n] = 1
        # st.session_state.run_bingo_extraction_key_triggered = False # 다음번 버튼 클릭을 위해 리셋

        max_bingo_val = len(lines) if lines else 1
        
        # st.session_state에 없는 경우를 대비한 초기화
        if num_input_key_at_least_n not in st.session_state:
            st.session_state[num_input_key_at_least_n] = 1
            
        min_bingo_lines = st.number_input(
            "추출할 최소 빙고 라인 수 (N)",
            min_value=1, 
            max_value=max_bingo_val,
            value=st.session_state[num_input_key_at_least_n], # 세션에서 값 로드
            step=1, 
            key=num_input_key_at_least_n, # 위젯 자체의 키
            # on_change 콜백을 사용하여 세션 상태 업데이트
            on_change=lambda: st.session_state.update({num_input_key_at_least_n: st.session_state[num_input_key_at_least_n]})
        )

        if st.button("N개 이상 빙고 달성자 목록 보기", key="view_multi_bingo_winners_key"):
            if user_count: # 빙고 달성자가 있는 경우에만 실행
                # min_bingo_lines는 위 number_input에서 최신 값으로 사용됨
                current_min_lines = st.session_state[num_input_key_at_least_n]
                winners_n_bingo = {uid: count_val for uid, count_val in user_count.items() if count_val >= current_min_lines}
                
                if winners_n_bingo:
                    df_multi_bingo = pd.DataFrame(list(winners_n_bingo.items()), columns=['UID', '달성 라인 수'])
                    df_multi_bingo = df_multi_bingo.sort_values(by=['달성 라인 수', 'UID'], ascending=[False, True]).reset_index(drop=True)
                    
                    st.write(f"#### {current_min_lines}개 이상 빙고 달성자 ({len(df_multi_bingo)}명)")
                    st.dataframe(df_multi_bingo)
                    
                    csv_multi_bingo = df_multi_bingo.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        label=f"{current_min_lines}개 이상 빙고 달성자 CSV 다운로드",
                        data=csv_multi_bingo,
                        file_name=f"bingo_winners_{current_min_lines}_or_more_lines.csv",
                        mime="text/csv",
                        key="download_multi_bingo_winners_csv_key" # 다운로드 버튼의 고유 키
                    )
                else:
                    st.info(f"{current_min_lines}개 이상의 빙고를 달성한 사용자가 없습니다.")
            else: # user_count가 비어있는 경우
                 st.info("빙고를 달성한 사용자가 없어 N개 이상 빙고 달성자 목록을 생성할 수 없습니다.")

# 세션 상태 초기화 관련 (선택적, 예시로 빙고 크기 변경 시 일부 값 리셋)
# if "previous_bingo_size" not in st.session_state:
#    st.session_state.previous_bingo_size = n
# elif st.session_state.previous_bingo_size != n:
#    # 빙고 크기가 변경되면 관련 세션 값들 초기화 (예: number_input 값)
#    num_input_key_at_least_n_old = f"min_bingo_lines_input_{st.session_state.previous_bingo_size}"
#    if num_input_key_at_least_n_old in st.session_state:
#        st.session_state[num_input_key_at_least_n_old] = 1 # 또는 del st.session_state[...]
#    st.session_state.previous_bingo_size = n
#    # 셀 선택도 초기화 필요하면 여기에 로직 추가

        
# ------------------------------------------------------------------------------
# J. 특정 열(컬럼) 삭제하기
# ------------------------------------------------------------------------------
st.subheader("8) 특정 열(컬럼) 삭제하기")

column_delete_target = st.selectbox(
    "열 삭제할 CSV를 선택하세요",
    ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values()),
    key="col_delete_select" # key 추가
)

use_header_for_delete = st.checkbox("CSV 첫 행을 헤더로 사용하기 (체크 시, 첫 행을 컬럼명으로 간주)", key="col_delete_header")

if column_delete_target != "--- 선택하세요 ---":
    orig_key_delete = [k for k, v in st.session_state["file_names"].items() if v == column_delete_target][0]
    df_for_delete = st.session_state["csv_dataframes"][orig_key_delete]
    df_temp_delete = df_for_delete.copy() # 원본 보존을 위해 복사본 사용

    if use_header_for_delete and not df_temp_delete.empty: # 비어있지 않은 경우에만 헤더 처리
        df_temp_delete.columns = df_temp_delete.iloc[0].astype(str) # 컬럼명을 문자열로 강제
        df_temp_delete = df_temp_delete[1:].reset_index(drop=True)
        # 헤더로 사용된 첫 행이 실제 데이터가 아니라면, df_for_delete에서 이를 반영해야 할 수도 있음
        # 현재는 df_temp_delete만 변경하고, 저장 시 df_temp_delete를 저장함.

    # 컬럼명이 숫자일 경우 문자열로 변환하여 표시 (multiselect에서 문제 방지)
    columns_list_delete = [str(col) for col in df_temp_delete.columns.tolist()]
    
    selected_cols_to_delete = st.multiselect(
        "삭제할 열을 선택하세요",
        options=columns_list_delete,
        key="col_delete_multiselect" # key 추가
    )

    if st.button("열 삭제 실행", key="col_delete_run"): # key 추가
        if not selected_cols_to_delete:
            st.warning("최소 한 개 이상의 열을 선택해 주세요.")
        else:
            # 선택된 컬럼명(문자열)을 실제 DataFrame의 컬럼 타입과 맞춰야 할 수 있음
            # 여기서는 df_temp_delete.columns가 이미 문자열로 변환되었거나,
            # pd.DataFrame.drop이 컬럼명 타입에 너그럽다고 가정
            df_after_delete = df_temp_delete.drop(columns=selected_cols_to_delete, errors="ignore")
            
            # 만약 use_header_for_delete가 True였다면, 삭제된 헤더를 다시 첫 행으로 돌려놓을지,
            # 아니면 헤더가 없는 상태로 저장할지 결정 필요.
            # 현재 코드는 헤더가 적용된 상태(첫 행이 데이터로 취급됨)에서 열 삭제 후 저장.
            # 사용자가 헤더를 포함하여 다운로드 받길 원한다면 추가 작업 필요.
            # 여기서는 헤더 사용 여부와 관계없이 최종 df_after_delete를 저장.

            new_file_name_deleted = f"{column_delete_target}_cols_removed.csv"
            save_to_session_and_download(df_after_delete, new_file_name_deleted)
            st.success(f"선택한 열이 삭제된 파일 '{new_file_name_deleted}'이(가) 생성되었습니다.")


# ------------------------------------------------------------------------------
# K. N개로 분할하기
# ------------------------------------------------------------------------------
st.subheader("9) 파일 분할하기")

split_target = st.selectbox(
    "분할할 CSV를 선택하세요",
    ["--- 선택하세요 ---"] + list(st.session_state["file_names"].values()),
    key="split_select" # key 추가
)
n_parts = st.number_input("몇 개로 분할할까요?", min_value=2, value=2, step=1, key="split_n_parts") # key 추가

if st.button("파일 분할 실행", key="split_run"): # key 추가
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
            n_parts = total_rows # 분할 개수를 최대 행 수로 조정

        # 분할 로직
        # n_parts가 0이 되는 경우 방지 (위에서 total_rows > 0 보장)
        base_chunk = total_rows // n_parts
        remainder = total_rows % n_parts

        st.write(f"총 행 수: {total_rows}, 분할 개수: {n_parts}")
        if total_rows > 0: # 실제로 분할할 내용이 있을 때만 상세 정보 표시
             st.write(f"각 파트 기본 크기: {base_chunk}, 나머지 분배 파트 수: {remainder}")


        start_idx = 0
        for i in range(n_parts):
            chunk_size = base_chunk + (1 if i < remainder else 0)
            if chunk_size == 0: # 분할 결과 빈 파일이 생기는 경우 건너뛰기 (이론상 발생 안 함)
                continue
            
            end_idx = start_idx + chunk_size
            df_chunk = df_to_split.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # 파일명에서 확장자 분리
            name_part, ext_part = os.path.splitext(split_target)
            part_file_name = f"{name_part}_part{i+1}{ext_part}"
            
            save_to_session_and_download(df_chunk, part_file_name)
            st.info(f"{part_file_name} : 행 {start_idx+1} ~ {end_idx} (총 {chunk_size}행) 생성 완료") # 사용자 친화적 인덱스
            start_idx = end_idx

# ------------------------------------------------------------------------------
# L. 사용자 지정 파일 순서 및 매트릭스 출력 (실행 버튼 및 출력 형식 선택 추가)
# ------------------------------------------------------------------------------
import os # 파일명 처리 위해 추가

st.subheader("10) 사용자 지정 파일 순서 및 매트릭스 출력")

all_available_files = list(st.session_state["file_names"].values()) # 변수명 변경
if not all_available_files:
    st.warning("업로드된 파일이 없습니다.")
else:
    st.markdown("**매트릭스에 사용할 파일 순서를 직접 지정해주세요.**")
    
    # ordered_file_count의 value를 사용 가능한 파일 수 내에서 기본값 설정
    default_file_count = min(2, len(all_available_files)) if len(all_available_files) > 0 else 1

    ordered_file_count = st.number_input("매트릭스에 사용할 파일 개수", 
                                           min_value=1, max_value=len(all_available_files), 
                                           value=default_file_count, step=1, key="matrix_file_count")
    
    ordered_files_for_matrix = [] # 변수명 변경
    # 파일 선택 UI 개선: 이전에 선택된 파일을 다음 선택지에서 제외 (선택 사항, 구현 복잡도 증가)
    # 현재는 중복 선택 가능하며, 아래에서 중복 에러 처리
    temp_available_files = list(all_available_files) # 복사본 사용

    for i in range(int(ordered_file_count)):
        # 각 selectbox에 고유한 key 부여
        # 이전 선택을 고려하여 다음 selectbox의 기본값을 다르게 설정 (고급 기능)
        # 여기서는 기본값 없이 사용자가 직접 선택하도록 둠
        file_sel = st.selectbox(f"{i+1}번째 파일 선택", temp_available_files, key=f"order_matrix_{i}")
        ordered_files_for_matrix.append(file_sel)
        # (선택적) 선택된 파일은 다음 드롭다운에서 제거 또는 비활성화 - 현재 미구현
        # if file_sel in temp_available_files:
        #     temp_available_files.remove(file_sel) # 이렇게 하면 다음 selectbox의 옵션이 줄어듦
    
    if len(set(ordered_files_for_matrix)) < len(ordered_files_for_matrix) and ordered_file_count > 0 :
        st.error("같은 파일이 여러 번 선택되었습니다. 각 순서에는 서로 다른 파일을 선택해주세요.")
    else:
        st.markdown("### 출력 형식 선택")
        representation_option = st.radio(
            "어떻게 결과를 표시할까요?",
            ("절대값 (비율)", "절대값", "비율"), # 순서 변경: 절대값(비율)을 기본으로
            key="representation_option_matrix" # key 추가
        )
        
        if st.button("매트릭스 생성하기", key="matrix_run"): # key 추가
            uid_sets_matrix = {} # 변수명 변경
            for fname_matrix in ordered_files_for_matrix:
                # 현재 파일명(fname_matrix)으로 오리지널 키 찾기
                orig_key_matrix = [k for k, v in st.session_state["file_names"].items() if v == fname_matrix][0]
                uid_sets_matrix[fname_matrix] = get_uid_set(orig_key_matrix)
            
            # (A) 파일 간 교집합 매트릭스
            matrix_data = [] # 변수명 변경
            header_row = [""] + ordered_files_for_matrix # 변수명 변경
            matrix_data.append(header_row)
            
            for i, file_i_name in enumerate(ordered_files_for_matrix):
                row_data = [file_i_name] # 변수명 변경
                base_count_i = len(uid_sets_matrix[file_i_name])
                for j, file_j_name in enumerate(ordered_files_for_matrix):
                    if j < i: # 대각선 아래쪽은 빈 칸 (또는 위와 동일 값 - 선택)
                        row_data.append("") 
                    else:
                        inter_count_ij = len(uid_sets_matrix[file_i_name].intersection(uid_sets_matrix[file_j_name]))
                        
                        if representation_option == "절대값":
                            cell_val_str = str(inter_count_ij)
                        elif representation_option == "비율":
                            # file_i 기준 비율 (대각선은 항상 100%)
                            ratio_val = (round(inter_count_ij / base_count_i * 100, 1) if base_count_i > 0 else 0.0)
                            cell_val_str = f"{ratio_val}%"
                        else:  # "절대값 (비율)"
                            ratio_val = (round(inter_count_ij / base_count_i * 100, 1) if base_count_i > 0 else 0.0)
                            cell_val_str = f"{inter_count_ij} ({ratio_val}%)"
                        row_data.append(cell_val_str)
                matrix_data.append(row_data)
            
            matrix_df_display = pd.DataFrame(matrix_data[1:], columns=matrix_data[0]) # 변수명 변경
            st.write("### 파일 간 교집합 매트릭스 (i행 파일 기준)")
            st.dataframe(matrix_df_display)
            save_to_session_and_download(matrix_df_display, "pairwise_intersection_matrix.csv")
            
            # (B) 잔존율 매트릭스
            retention_matrix_data = [] # 변수명 변경
            header_row_retention = [""] + ordered_files_for_matrix # 변수명 변경
            retention_matrix_data.append(header_row_retention)
            
            for i, file_i_ret_name in enumerate(ordered_files_for_matrix):
                row_data_retention = [file_i_ret_name] # 변수명 변경
                base_count_ret_i = len(uid_sets_matrix[file_i_ret_name])
                
                current_intersection_set = uid_sets_matrix[file_i_ret_name].copy() # 각 행 시작 시 초기화
                
                for j, file_j_ret_name in enumerate(ordered_files_for_matrix):
                    if j < i: # 대각선 아래쪽은 빈 칸
                        row_data_retention.append("")  
                    else:
                        # i부터 j까지의 파일들과 순차적 교집합
                        # (j > i 일때만 file_j_ret_name과 교집합 추가)
                        if j > i : # 이전 교집합 결과에 현재 파일(j)의 UID set을 교집합
                           current_intersection_set = current_intersection_set.intersection(uid_sets_matrix[file_j_ret_name])
                        
                        # j == i 일때는 current_intersection_set은 file_i_name의 UID set 그대로임

                        abs_val_ret = len(current_intersection_set)
                        rate_ret = round(abs_val_ret / base_count_ret_i * 100, 1) if base_count_ret_i > 0 else 0.0
                        
                        if representation_option == "절대값":
                            cell_val_ret_str = str(abs_val_ret)
                        elif representation_option == "비율":
                            cell_val_ret_str = f"{rate_ret}%"
                        else:  # "절대값 (비율)"
                            cell_val_ret_str = f"{abs_val_ret} ({rate_ret}%)"
                        row_data_retention.append(cell_val_ret_str)
                retention_matrix_data.append(row_data_retention)
            
            retention_df_display = pd.DataFrame(retention_matrix_data[1:], columns=retention_matrix_data[0]) # 변수명 변경
            st.write(f"### 잔존율 매트릭스 ({ordered_files_for_matrix[0]} 기준 시작)")
            st.dataframe(retention_df_display)
            save_to_session_and_download(retention_df_display, "retention_matrix.csv")

# ------------------------------------------------------------------------------
# M. 벤 다이어그램 생성
# ------------------------------------------------------------------------------
st.subheader("11) 벤 다이어그램 생성") # 번호 수정

selected_for_venn = st.multiselect(
    "벤 다이어그램에 사용할 CSV 파일 선택 (최대 3개)",
    list(st.session_state["file_names"].values()),
    key="venn_select" # key 추가
)

if st.button("벤 다이어그램 생성하기", key="venn_run"): # key 추가
    if not selected_for_venn:
        st.error("최소 1개의 CSV 파일을 선택해주세요.")
    elif len(selected_for_venn) > 3:
        st.error("벤 다이어그램은 현재 1~3개의 집합만 지원합니다.")
    else:
        venn_sets_dict = {} # 변수명 변경
        venn_set_labels = [] # 순서대로 레이블 저장
        for fname_venn in selected_for_venn:
            orig_key_venn = [k for k, v in st.session_state["file_names"].items() if v == fname_venn][0]
            venn_sets_dict[fname_venn] = get_uid_set(orig_key_venn)
            venn_set_labels.append(fname_venn) # 선택된 순서대로 레이블 저장
        
        # matplotlib_venn 임포트는 함수 내부 또는 상단에 한 번만
        from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles 
        import matplotlib.patches as mpatches

        fig_venn, ax_venn = plt.subplots(figsize=(8,8)) # 변수명 변경 및 figsize 조정
        
        plt.title(f"벤 다이어그램 ({len(selected_for_venn)} 집합)", fontsize=14)

        if len(selected_for_venn) == 1:
            set1_venn = venn_sets_dict[venn_set_labels[0]]
            count1 = len(set1_venn)
            
            # venn2를 사용하여 하나의 원 그리기 (꼼수)
            # venn2([set1_venn, set()], set_labels=(venn_set_labels[0], ''))
            # v.get_patch_by_id('10').set_alpha(0.5)
            # v.get_patch_by_id('10').set_color('skyblue')
            # if v.get_label_by_id('10'): v.get_label_by_id('10').set_text(f'{venn_set_labels[0]}\n({count1})')
            # if v.get_label_by_id('01'): v.get_label_by_id('01').set_text('')
            # if v.get_label_by_id('11'): v.get_label_by_id('11').set_text('')
            # 위 방법 대신 직접 원 그리기
            circle = plt.Circle((0.5, 0.5), 0.4, color="skyblue", alpha=0.7)
            ax_venn.add_patch(circle)
            ax_venn.text(0.5, 0.5, f"{venn_set_labels[0]}\n({count1})", 
                         horizontalalignment='center', verticalalignment='center', 
                         fontsize=12, fontweight='bold')
            ax_venn.set_xlim(0, 1); ax_venn.set_ylim(0, 1)
            ax_venn.axis('off')
            # 범례
            legend_patch1 = mpatches.Patch(color='skyblue', label=f"{venn_set_labels[0]} ({count1})")
            ax_venn.legend(handles=[legend_patch1], loc="best", fontsize=10)


        elif len(selected_for_venn) == 2:
            set1_venn = venn_sets_dict[venn_set_labels[0]]
            set2_venn = venn_sets_dict[venn_set_labels[1]]
            v = venn2([set1_venn, set2_venn], 
                      set_labels=tuple(venn_set_labels), 
                      ax=ax_venn,
                      set_colors=('skyblue', 'lightcoral'), alpha = 0.7)
            # 각 영역 레이블 폰트 크기 조정 (선택적)
            for text_id in ['10', '01', '11']:
                if v.get_label_by_id(text_id):
                    v.get_label_by_id(text_id).set_fontsize(10)
            venn2_circles([set1_venn, set2_venn], linestyle='dashed', linewidth=1, color='grey', ax=ax_venn)


        elif len(selected_for_venn) == 3:
            set1_venn = venn_sets_dict[venn_set_labels[0]]
            set2_venn = venn_sets_dict[venn_set_labels[1]]
            set3_venn = venn_sets_dict[venn_set_labels[2]]
            v = venn3([set1_venn, set2_venn, set3_venn], 
                      set_labels=tuple(venn_set_labels), 
                      ax=ax_venn,
                      set_colors=('skyblue', 'lightcoral', 'lightgreen'), alpha = 0.7)
            # 각 영역 레이블 폰트 크기 조정 (선택적)
            for text_id in ['100', '010', '001', '110', '101', '011', '111']:
                if v.get_label_by_id(text_id):
                    v.get_label_by_id(text_id).set_fontsize(9)
            venn3_circles([set1_venn, set2_venn, set3_venn], linestyle='dashed', linewidth=1, color='grey', ax=ax_venn)
        
        # 범례는 2개, 3개 집합일 때 matplotlib_venn이 자동으로 생성하지 않으므로, set_labels로 대체.
        # 필요하다면 mpatches를 사용하여 수동으로 생성 가능.
        # (위의 1개 집합 경우처럼)

        # plt.tight_layout() # tight_layout은 Streamlit에서 자동으로 처리되는 경우가 많음
        st.pyplot(fig_venn) # 변경된 변수명 사용
