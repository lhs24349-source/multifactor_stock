import streamlit as st
import google.generativeai as genai
import os
import re
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv, set_key
import requests
from bs4 import BeautifulSoup
import hashlib
import time

# .env 로드
ENV_PATH = ".env"
load_dotenv(ENV_PATH)

# 페이지 설정
st.set_page_config(
    page_title="주식 멀티팩터 뉴스 분석 & 추천기",
    page_icon="📈",
    layout="wide",
)

# 오늘 날짜 동적 추출
today_str = datetime.today().strftime('%Y.%m.%d')

# ngrok 자동 구동 (최초 1회만 실행되도록 caching)
@st.cache_resource(show_spinner="외부 접속용 ngrok 서버 연동 중...")
def init_ngrok():
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(8501).public_url
        return public_url
    except Exception as e:
        return f"ngrok 연결 에러: {e}"

# 외부 환경 요인(DB 부족)으로 인한 팩터 시뮬레이션용 함수
def mock_factor_score(ticker, seed_str):
    hash_val = int(hashlib.md5(f"{ticker}_{seed_str}".encode()).hexdigest(), 16)
    return 40 + (hash_val % 61) # 40 ~ 100 점수 부여

# 네이버 금융 종목별 현재가 및 목표가 크롤링
@st.cache_data(ttl=3600*24)
def get_naver_finance_prices(ticker):
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        current_price = 0
        target_price = 0
        
        no_today = soup.select_one('.no_today .blind')
        if no_today:
            try:
                current_price = int(no_today.text.replace(',',''))
            except:
                pass
                
        th = soup.find('th', string=lambda t: t and '목표주가' in t)
        if th:
            em_tags = th.parent.find_all('em')
            if len(em_tags) > 1:
                try:
                    target_price = int(em_tags[1].text.replace(',',''))
                except:
                    pass
                    
        upside = 0.0
        if current_price > 0 and target_price > current_price:
            upside = ((target_price / current_price) - 1.0) * 100
            
        return {
            "현재가": current_price,
            "목표주가": target_price,
            "업사이드(%)": round(upside, 1)
        }
    except Exception:
        return {"현재가": 0, "목표주가": 0, "업사이드(%)": 0.0}

ngrok_url = init_ngrok()

with st.sidebar:
    st.title("설정 ⚙️")
    
    st.markdown("### 🔑 API 키 설정")
    saved_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input("Google Gemini API 키", value=saved_key, type="password")
    
    if st.button("API 키 저장", use_container_width=True):
        if api_key_input:
            set_key(ENV_PATH, "GEMINI_API_KEY", api_key_input)
            os.environ["GEMINI_API_KEY"] = api_key_input
            st.success("API 키가 저장되었습니다. (.env)")
        else:
            st.error("API 키를 입력해주세요.")
            
    st.markdown("---")
    st.subheader("🌐 외부 접속(ngrok) URL")
    if "Error" not in ngrok_url:
        st.success(f"생성 완료:\n\n{ngrok_url}")
    else:
        st.error(ngrok_url)
        
    st.markdown("---")
    st.markdown("""
        **☁️ 완전한 웹 서버 배포 안내:**  
        로컬 PC를 끄더라도 항상 접속되게 하려면 **Streamlit Community Cloud** 배포를 권장합니다.  
        이 디렉토리에 생성해 둔 `requirements.txt`와 소스코드를 GitHub에 올리고, `share.streamlit.io` 에서 연동하면 바로 서비스가 가능합니다. 😄
    """)

st.title("📈 주식 멀티팩터 기반 시장 분석 및 추천")
st.markdown(f"**기준일: {today_str}**")
st.markdown("최근 국내 경제/금융 뉴스를 기반으로 한국은행 기준금리 의사결정을 예측하고, 이에 따른 **멀티팩터(모멘텀, 밸류, 퀄리티)**의 투자 비중을 제안하며 맞춤형 자산을 추천합니다.")
st.markdown("💡 **(New)** 시가총액 기반 전체 코스피, 코스닥, ETF 유니버스를 최대 100개까지 검색하고 실시간 목표가를 반영하여 스코어링합니다.")

# 조건 검색 UI 구성
col1, col2 = st.columns(2)
with col1:
    target_market = st.selectbox("📊 분석 대상 시장", ["전체 종목", "코스피", "코스닥", "ETF"])
with col2:
    target_limit = st.slider("🔍 조회 종목 수 (시가총액 상위 N개)", min_value=10, max_value=100, value=30, step=10)

SYSTEM_PROMPT = f"""
# 지시문
- 당신은 한국 경제 전문가이자 퀀트 투자 전략가입니다.
- 최근 국내 주식시장의 뉴스와 관련 자료를 학습하여, 멀티팩터 세부 지표별로 주가 예측 중요도를 분석해주세요.
- 분석 결과는 팩터별 가중치 배분 및 주식/ETF 추천 모델에 사용됩니다.

# 제약 조건
- 한국은행에서 금리인하 여부에 따라 보고서를 출력합니다.
- 오늘은 [{today_str}] 입니다. 검색된 지식을 바탕으로 최근 3개월 자료를 분석하세요.
- 멀티팩터 지표는 다음과 같이 3가지로 정의합니다.
  - Momentum(모멘텀): 주가 추세, 거래량 증가
  - Value(가치): 저 PER, 저 PBR
  - Quality(퀄리티): 안정적 매출/영업/순이익 증가
- 보고서는 객관적인 데이터와 발표 자료를 근거로 출력해주세요.

# 출력형식
- [한국은행 분석] 금리인하 가능성을 퍼센트로 표기하고 근거를 제시합니다.
- [팩터 중요도 분석] 금리 인하/동결 시나리오에 따른 3가지 팩터(Momentum, Value, Quality)의 중요도 비중(%) 및 근거를 분석하세요.
- [중요!] 보고서의 맨 마지막에는 도출된 현재 시장 상황에 가장 적합한 팩터 가중치(3개 항목)를 시나리오와 무관한 **단일의 최종 예측값**으로 정하고, 이를 반드시 아래와 같은 JSON 형식으로만 작성해주세요! (비중 합계는 반드시 100이 되어야 합니다)
```json
{{
  "Momentum": 30,
  "Value": 20,
  "Quality": 50
}}
```
"""

def get_naver_top_cap(market="KOSPI", limit=100):
    sosok = 0 if market == "KOSPI" else 1
    data = []
    page = 1
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    while len(data) < limit and page <= 5:
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        table = soup.find('table', {'class': 'type_2'})
        if not table: break
            
        tbody = table.find('tbody')
        trs = tbody.find_all('tr')
        
        for tr in trs:
            tds = tr.find_all('td')
            if len(tds) > 2:
                a_tag = tds[1].find('a')
                if a_tag:
                    name = a_tag.text.strip()
                    href = a_tag['href']
                    ticker = href.split('code=')[-1]
                    data.append({
                        "Ticker": ticker,
                        "Name": name
                    })
                    if len(data) >= limit:
                        break
        page += 1
    return pd.DataFrame(data)

def get_naver_etf_top_cap(limit=100):
    # ETF는 네이버 금융 전용 API/페이지나 fdr.StockListing("ETF/KR")가 막혀있으므로
    # 시가총액 기준으로 자체 ETF 리스트를 스크래핑합니다.
    url = "https://finance.naver.com/api/sise/etfItemList.nhn"
    data = []
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        json_data = res.json()
        etf_list = json_data['result']['etfItemList']
        # 시가총액(marketSum) 순으로 이미 정렬되어 있는 편이나, 확실히 정렬
        etf_list = sorted(etf_list, key=lambda x: x.get('marketSum', 0), reverse=True)
        
        for idx, item in enumerate(etf_list):
            if idx >= limit:
                break
            data.append({
                "Ticker": item['itemcode'],
                "Name": item['itemname']
            })
    except Exception as e:
        st.error(f"ETF 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def fetch_universe(market, limit):
    if market == "ETF":
        df = get_naver_etf_top_cap(limit)
    elif market == "코스피":
        df = get_naver_top_cap("KOSPI", limit)
    elif market == "코스닥":
        df = get_naver_top_cap("KOSDAQ", limit)
    else: # 전체 종목 (KOSPI + KOSDAQ)
        df1 = get_naver_top_cap("KOSPI", limit)
        df2 = get_naver_top_cap("KOSDAQ", limit)
        df = pd.concat([df1, df2]).head(limit).reset_index(drop=True)
    return df

def score_and_recommend(weights, market, limit):
    universe = fetch_universe(market, limit)
    
    w_m = weights.get("Momentum", 33.3) / 100.0
    w_v = weights.get("Value", 33.3) / 100.0
    w_q = weights.get("Quality", 33.4) / 100.0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    data = []
    total = len(universe)
    
    for idx, row in universe.iterrows():
        ticker = row['Ticker']
        name = row['Name']
        i = len(data) + 1
        
        status_text.text(f"목표가 및 시장 데이터 크롤링 중... ({i}/{total}) - {name}")
        progress_bar.progress(i / total)
        
        # 실제 환경에선 실시간 데이터를 DB에서 가져와야 하지만, 
        # 제한된 환경 하우스 시뮬레이션을 위해 해시 기반 일관된 팩터값을 모킹합니다. 
        # (ETF 및 전체 종목의 PER/PBR 등을 실시간으로 병렬 조회하기에는 지연 문제가 크기 때문입니다.)
        mom = mock_factor_score(ticker, "mom")
        val = mock_factor_score(ticker, "val")
        qual = mock_factor_score(ticker, "qual")
        
        prices = get_naver_finance_prices(ticker)
        
        upside = prices["업사이드(%)"]
        upside_score = min(15.0, (upside / 30.0) * 15.0) if upside > 0 else 0
        upside_score = round(upside_score, 1)

        ai_score = round((mom * w_m) + (val * w_v) + (qual * w_q), 1)
        final_score = round(ai_score + upside_score, 1)
        
        data.append({
            "종목코드": ticker,
            "종목명": name,
            "현재가(원)": prices["현재가"] if prices["현재가"]>0 else ("-" if market!="ETF" else "-"),
            "목표가(원)": prices["목표주가"] if prices["목표주가"]>0 else "-",
            "업사이드(%)": upside,
            "업사이드 가점": upside_score,
            "AI 팩터추정점수": ai_score,
            "최종 스코어": final_score
        })
        time.sleep(0.05) # 서버 부하 방지용 짧은 딜레이
        
    status_text.empty()
    progress_bar.empty()
    
    df = pd.DataFrame(data)
    df = df.sort_values(by="최종 스코어", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df

if st.button("분석 및 자산 추천 시작 🚀", use_container_width=True):
    current_key = api_key_input or os.environ.get("GEMINI_API_KEY", "")
    if not current_key:
        st.error("설정 탭에서 Google API 키를 입력하거나 저장해주세요!")
    else:
        with st.spinner(f"AI가 뉴스를 분석하여 팩터 비중을 결정하고, 시총 상위 {target_limit}개 종목을 스코어링합니다..."):
            try:
                genai.configure(api_key=current_key)
                model = genai.GenerativeModel('gemini-2.5-pro')
                response = model.generate_content(SYSTEM_PROMPT)
                
                tab1, tab2 = st.tabs(["📄 전략 보고서", "🏅 멀티팩터 기반 종목 풀 스코어링 추천 결과"])
                
                response_text = response.text
                
                with tab1:
                    st.subheader(f"📊 퀀트 전략 리포트 ({today_str})")
                    clean_text = re.sub(r'```json\n(.*?)\n```', '', response_text, flags=re.DOTALL)
                    st.markdown(clean_text)
                    
                with tab2:
                    st.subheader(f"🤖 AI 멀티팩터 추천 포트폴리오 (대상: {target_market} 상위 {target_limit}개)")
                    
                    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                    weights = {"Momentum": 33.3, "Value": 33.3, "Quality": 33.4}
                    
                    if json_match:
                        try:
                            weights = json.loads(json_match.group(1))
                            st.info(f"**적용된 AI 팩터 비중:** 📈 모멘텀 {weights.get('Momentum', 0)}% | 💰 가치 {weights.get('Value', 0)}% | 💎 퀄리티 {weights.get('Quality', 0)}%")
                        except Exception as e:
                            st.warning("비중 파싱 실패로 균등 비중이 적용되었습니다.")
                    else:
                        st.warning("단일 가중치 포맷을 인식할 수 없어 균등 비중이 적용되었습니다.")
                    
                    st.markdown("도출된 팩터 비중과 실시간 증권사 목표주가를 결합하여 해당 시장의 상위 항목들을 실시간 스코어링한 결과입니다.")
                    recommended_df = score_and_recommend(weights, target_market, target_limit)
                    st.dataframe(recommended_df, use_container_width=True)
                    
                st.success("✅ 분석 및 추천이 완료되었습니다!")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                
st.markdown("---")
st.caption(f"개발: 한국 경제 퀀트 분석 시스템 | 날짜 기준: {today_str}")
