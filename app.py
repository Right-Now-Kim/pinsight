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

# (여기에 E.조합, F.N번이상, G.중복제거, H.랜덤추출, I.빙고, J.열삭제, K.파일분할, L.매트릭스, M.벤다이어그램 코드 삽입)
# 각 기능의 파일 선택 부분은 list(st.session_state["file_names"].values())를 사용하고,
# UID set을 가져올 때는 get_uid_set_from_display_name(선택된_표시명)을 사용합니다.
# 결과 저장 시에는 save_result_to_session_and_offer_download(결과df, "기본파일명.csv")를 사용합니다.
# 모든 st 위젯에는 고유한 key를 부여해야 합니다.

st.markdown("---") # CSV 기능과 웹툰 추출 기능 구분

# ------------------------------------------------------------------------------
# N. 카카오페이지 웹툰 업데이트 일자 추출 (이전 답변의 최종 코드 버전 사용)
# ------------------------------------------------------------------------------
# (이전 답변에서 제공된 카카오페이지 웹툰 업데이트 일자 추출 기능의 최종 코드를 여기에 그대로 붙여넣으세요.)
# st.header, st.text_input, st.button, @st.cache_data 데코레이터가 있는 함수,
# get_update_dates_for_series_internal 함수, 그리고 메인 실행 버튼 로직 전체입니다.
# key 값들이 위의 CSV 기능들과 중복되지 않도록 주의하세요. (예: _kp 접미사 추가)

st.header("🌐 카카오페이지 웹툰 정보 추출") # 헤더
st.subheader("업데이트 일자 추출")

kakaopage_series_ids_input_kp = st.text_input( # key 변경
    "카카오페이지 작품 ID를 쉼표(,)로 구분하여 입력하세요 (예: 59782511, 12345678)",
    key="kakaopage_ids_input_main_kp"
)

log_container_kp = st.container()
process_logs_kp = []

# (get_update_dates_for_series_internal 함수는 이전 답변의 것을 여기에 복사)
# (get_update_dates_for_series_cached_wrapper 함수는 이전 답변의 것을 여기에 복사)
# (웹툰 추출 실행 버튼 로직은 이전 답변의 것을 여기에 복사, key는 _kp 등으로 변경)

# 예시로 get_update_dates_for_series_internal 함수와 wrapper, 버튼 로직만 가져옴
# 실제로는 이 함수들을 정의하고 사용해야 합니다.

# --- 실제 스크래핑 및 캐싱 함수 (이전 답변의 코드를 여기에 삽입) ---
def get_update_dates_for_series_internal(series_id, driver, log_callback_ui):
    # ... (이전 답변의 get_update_dates_for_series_internal 함수 내용 전체) ...
    # (이 함수는 Streamlit UI 객체를 직접 호출하지 않도록 log_callback_ui를 사용)
    url = f"https://page.kakao.com/content/{series_id}"
    log_callback_ui(f"ID {series_id}: 스크래핑 시작. URL: {url}")
    driver.get(url)
    update_dates = []
    
    try:
        WebDriverWait(driver, 20).until( 
            EC.presence_of_element_located((By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]"))
        )
        log_callback_ui(f"ID {series_id}: 초기 회차 목록 컨테이너(ul) 로드 확인.")
        time.sleep(3) 

        max_scroll_attempts = 25 
        no_new_content_streak = 0
        max_no_new_content_streak = 3
        
        initial_items_count = len(driver.find_elements(By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]/li"))
        log_callback_ui(f"ID {series_id}: '더보기' 전, 초기 감지된 회차 아이템 수: {initial_items_count}")

        for attempt in range(max_scroll_attempts):
            items_before_click_elements = driver.find_elements(By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]/li//div[contains(@class, 'font-x-small1')]//span[@class='break-all align-middle'][1]")
            count_before_click = len(items_before_click_elements)

            try:
                load_more_button_xpath = "//ul[contains(@class, 'jsx-3287026398')]/following-sibling::div[1][.//img[@alt='아래 화살표']]"
                load_more_button = WebDriverWait(driver, 8).until( 
                    EC.element_to_be_clickable((By.XPATH, load_more_button_xpath))
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", load_more_button)
                time.sleep(0.6) 
                driver.execute_script("arguments[0].click();", load_more_button)
                log_callback_ui(f"ID {series_id}: '더보기' 버튼 클릭 ({attempt + 1}/{max_scroll_attempts}).")
                time.sleep(2) 

                items_after_click_elements = driver.find_elements(By.XPATH, "//ul[contains(@class, 'jsx-3287026398')]/li//div[contains(@class, 'font-x-small1')]//span[@class='break-all align-middle'][1]")
                count_after_click = len(items_after_click_elements)

                if count_after_click == count_before_click:
                    no_new_content_streak += 1
                    log_callback_ui(f"ID {series_id}: 새 콘텐츠 변화 없음 ({no_new_content_streak}/{max_no_new_content_streak}). 현재까지 감지된 날짜 수: {count_after_click}.")
                    if no_new_content_streak >= max_no_new_content_streak:
                        log_callback_ui(f"ID {series_id}: 연속 {max_no_new_content_streak}회 새 콘텐츠 변화 없어 '더보기' 중단.")
                        break
                else:
                    no_new_content_streak = 0
                    log_callback_ui(f"ID {series_id}: 새 콘텐츠 로드됨. 현재까지 감지된 날짜 수: {count_after_click}.")
            
            except TimeoutException:
                log_callback_ui(f"ID {series_id}: '더보기' 버튼 타임아웃. 모든 회차 로드 완료로 간주.")
                break
            except NoSuchElementException:
                log_callback_ui(f"ID {series_id}: '더보기' 버튼을 더 이상 찾을 수 없음. 모든 회차 로드 완료로 간주.")
                break
            except ElementClickInterceptedException:
                log_callback_ui(f"ID {series_id}: '더보기' 버튼 클릭 가로채짐. 페이지 하단으로 스크롤 후 재시도.")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5) 
            except Exception as e_click_internal_error:
                log_callback_ui(f"ID {series_id}: '더보기' 버튼 클릭 중 예상치 못한 오류: {str(e_click_internal_error)[:100]}")
                break

        episode_list_items_xpath = "//ul[contains(@class, 'jsx-3287026398')]/li[contains(@class, 'list-child-item')]"
        list_items_elements = driver.find_elements(By.XPATH, episode_list_items_xpath)
        log_callback_ui(f"ID {series_id}: 최종적으로 스캔할 회차 아이템 요소 수: {len(list_items_elements)}.")
        if not list_items_elements:
            log_callback_ui(f"ID {series_id}: [경고] 최종 회차 아이템 요소를 하나도 찾지 못했습니다.")

        date_pattern = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")
        extracted_this_run = []
        for idx, item_element in enumerate(list_items_elements):
            try:
                date_span_xpath = ".//div[contains(@class, 'font-x-small1') and contains(@class, 'h-16pxr') and contains(@class, 'text-el-50')]/span[@class='break-all align-middle'][1]"
                date_span = item_element.find_element(By.XPATH, date_span_xpath)
                date_text = date_span.text.strip()
                if date_pattern.match(date_text):
                    extracted_this_run.append(date_text)
            except NoSuchElementException:
                pass 
            except Exception: 
                pass
        
        seen_dates = set()
        for d_item_internal in extracted_this_run:
            if d_item_internal not in seen_dates:
                update_dates.append(d_item_internal)
                seen_dates.add(d_item_internal)
        
        if not update_dates:
            log_callback_ui(f"ID {series_id}: [결과] 추출된 업데이트 날짜가 없습니다. (총 {len(extracted_this_run)}개의 날짜 형식 텍스트 중 유효한 날짜 0개)")
        else:
            log_callback_ui(f"ID {series_id}: [결과] {len(update_dates)}개의 고유한 업데이트 날짜 추출 완료.")
                
    except TimeoutException:
        log_callback_ui(f"ID {series_id}: [오류] 페이지의 주요 컨텐츠(회차 목록) 로드 시간 초과.")
    except Exception as e_global_scrape_internal_error:
        log_callback_ui(f"ID {series_id}: [오류] 스크래핑 중 예기치 않은 오류 발생: {str(e_global_scrape_internal_error)[:150]}")
    
    return update_dates

@st.cache_data(ttl=3600, show_spinner=False)
def get_update_dates_for_series_cached_wrapper(series_id, webdriver_options_dict):
    temp_logs_for_cache = []
    def append_log_for_cache_internal(message):
        temp_logs_for_cache.append(message)

    options = webdriver.ChromeOptions()
    for arg_name, arg_val in webdriver_options_dict.get("arguments", {}).items():
        if arg_val is None: options.add_argument(arg_name)
        else: options.add_argument(f"{arg_name}={arg_val}")
    for opt_name, opt_value in webdriver_options_dict.get("experimental_options", {}).items():
        options.add_experimental_option(opt_name, opt_value)

    driver_instance_cache_internal = None
    try:
        s_cache_internal = ChromeService(ChromeDriverManager().install())
        driver_instance_cache_internal = webdriver.Chrome(service=s_cache_internal, options=options)
        dates = get_update_dates_for_series_internal(series_id, driver_instance_cache_internal, append_log_for_cache_internal)
        return dates, temp_logs_for_cache
    except Exception as e_cache_internal:
        append_log_for_cache_internal(f"ID {series_id}: 캐시된 WebDriver 실행 중 심각한 오류: {str(e_cache_internal)}")
        return [], temp_logs_for_cache
    finally:
        if driver_instance_cache_internal:
            driver_instance_cache_internal.quit()

if st.button("업데이트 일자 추출 및 ZIP 다운로드", key="kakaopage_extract_button_main_kp"): # key 변경
    if not kakaopage_series_ids_input_kp:
        st.warning("작품 ID를 입력해주세요.")
    else:
        series_ids_list_kp = [id_str.strip() for id_str in kakaopage_series_ids_input_kp.split(',') if id_str.strip()]
        if not series_ids_list_kp:
            st.warning("유효한 작품 ID가 없습니다.")
        else:
            webdriver_options_dict_for_cache_final_kp = {
                "arguments": {
                    "--headless": None, "--no-sandbox": None, "--disable-dev-shm-usage": None,
                    "--disable-gpu": None,
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                    "--lang": "ko_KR", "--window-size=1920,1080"
                },
                "experimental_options": {"excludeSwitches": ['enable-logging']}
            }
            
            results_for_zip_kp = {}
            total_ids_kp = len(series_ids_list_kp)
            
            process_logs_kp.clear()
            log_container_kp.empty()
            log_display_area_kp = log_container_kp.expander("실시간 처리 로그 보기", expanded=True)
            
            progress_bar_placeholder_kp = st.empty()
            current_status_text_kp = st.empty()
            overall_start_time_kp = time.time()

            with st.spinner(f"총 {total_ids_kp}개의 작품 정보를 처리 중입니다... (각 ID 처리 시 웹페이지를 직접 방문합니다)"):
                progress_bar_kp = progress_bar_placeholder_kp.progress(0)
                
                for i, series_id_item_kp in enumerate(series_ids_list_kp):
                    start_time_item_kp = time.time()
                    current_progress_kp = (i + 1) / total_ids_kp
                    progress_bar_kp.progress(current_progress_kp)
                    current_status_text_kp.text(f"처리 중: {series_id_item_kp} ({i+1}/{total_ids_kp})")
                    
                    process_logs_kp.append(f"--- ID: {series_id_item_kp} 처리 시작 ---")
                    with log_display_area_kp: st.markdown(f"**ID: {series_id_item_kp} 처리 시작...**")
                    
                    dates_kp, item_logs_from_cache_kp = get_update_dates_for_series_cached_wrapper(series_id_item_kp, webdriver_options_dict_for_cache_final_kp)
                    
                    process_logs_kp.extend(item_logs_from_cache_kp)
                    with log_display_area_kp:
                        for log_msg_kp in item_logs_from_cache_kp:
                            if "[오류]" in log_msg_kp or "[경고]" in log_msg_kp: st.warning(log_msg_kp)
                            else: st.info(log_msg_kp)
                    
                    if dates_kp:
                        file_content_kp = "\n".join(dates_kp)
                        safe_series_id_kp = re.sub(r'[\\/*?:"<>|]', "_", series_id_item_kp)
                        filename_in_zip_kp = f"{safe_series_id_kp}_updates.txt"
                        results_for_zip_kp[filename_in_zip_kp] = file_content_kp
                        final_msg_kp = f"ID {series_id_item_kp}: [성공] {len(dates_kp)}개 업데이트 일자 추출 완료."
                        process_logs_kp.append(final_msg_kp)
                        with log_display_area_kp: st.success(final_msg_kp)
                    else:
                        final_msg_kp = f"ID {series_id_item_kp}: [실패] 업데이트 일자를 찾을 수 없거나 추출에 실패했습니다."
                        process_logs_kp.append(final_msg_kp)
                        with log_display_area_kp: st.error(final_msg_kp)
                    
                    end_time_item_kp = time.time()
                    process_logs_kp.append(f"ID {series_id_item_kp} 처리 소요 시간: {end_time_item_kp - start_time_item_kp:.2f}초")
                    process_logs_kp.append(f"--- ID: {series_id_item_kp} 처리 종료 ---\n")
                    with log_display_area_kp: st.markdown("---")
                    time.sleep(0.3)

            progress_bar_placeholder_kp.empty()
            current_status_text_kp.empty()
            overall_end_time_kp = time.time()
            process_logs_kp.insert(0, f"**카카오페이지 추출 전체 작업 완료! 총 소요 시간: {overall_end_time_kp - overall_start_time_kp:.2f}초**")

            if results_for_zip_kp:
                log_final_summary_kp = "모든 작품 처리 완료. ZIP 파일 생성 중..."
                process_logs_kp.append(log_final_summary_kp)
                # st.info(log_final_summary_kp) # expander에 표시되므로 중복 제거

                zip_buffer_kp = io.BytesIO()
                with zipfile.ZipFile(zip_buffer_kp, "w", zipfile.ZIP_DEFLATED) as zf_kp:
                    for filename_kp, content_kp in results_for_zip_kp.items():
                        zf_kp.writestr(filename_kp, content_kp.encode('utf-8'))
                zip_buffer_kp.seek(0)
                
                st.download_button(
                    label="추출된 업데이트 일자 ZIP 다운로드",
                    data=zip_buffer_kp,
                    file_name="kakaopage_webtoon_updates.zip",
                    mime="application/zip",
                    key="download_kakaopage_zip_main_v3"
                )
                process_logs_kp.append("ZIP 파일 생성 완료! 위 버튼으로 다운로드하세요.")
                st.success("카카오페이지 업데이트 일자 ZIP 파일 생성 완료! 다운로드 버튼을 이용하세요.")
            else:
                log_final_summary_kp = "추출된 데이터가 없어 ZIP 파일을 생성할 수 없습니다."
                process_logs_kp.append(log_final_summary_kp)
                st.warning(log_final_summary_kp)
            
            log_container_kp.empty()
            with st.expander("카카오페이지 추출 전체 상세 처리 로그 보기", expanded=False):
                for log_line_kp in process_logs_kp:
                    if "ID:" in log_line_kp and "시작" in log_line_kp: st.markdown(f"**{log_line_kp}**")
                    elif "[성공]" in log_line_kp: st.success(log_line_kp)
                    elif "[실패]" in log_line_kp or "[오류]" in log_line_kp or "[경고]" in log_line_kp : st.error(log_line_kp)
                    elif "소요 시간" in log_line_kp or "처리 종료" in log_line_kp: st.caption(log_line_kp)
                    elif "ZIP 파일" in log_line_kp or "총 소요 시간" in log_line_kp: st.info(log_line_kp)
                    else: st.markdown(f"`{log_line_kp}`")
# ------------------------------------------------------------------------------
# 앱 하단 정보 (선택 사항)
# ------------------------------------------------------------------------------
st.markdown("---")
st.caption("Pinsight Utility App by YourName/Organization (문의: ...)")
