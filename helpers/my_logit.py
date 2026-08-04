import numpy as np
import seaborn as sb
from IPython.display import display, Markdown
from pandas import DataFrame
from statsmodels.api import add_constant, Logit
from sklearn.metrics import(confusion_matrix, roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score)

from . import my_plot
from . import my_stats


#-----------------------------
# 로지스틱 모형 적합 함수 정의
#-----------------------------
def fit_model(data, y, summary=False):
    """
    statsmodels의 Logit을 이용해 이항 로지스틱 회귀 모델을 적합한다.

    종속변수 'y'를 제외한 나머지 모든 컬럼을 독립변수로 사용하며,
    절편(상수항)을 자동으로 추가한 뒤 최대우도추정(MLE)으로 회귀계수를 추정한다.
    종속변수는 0/1의 두 값만 가지는 이분형이어야 한다.

    Args:
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임
        y: 종속변수로 사용할 컬럼명. 'data'에 반드시 존재해야 하며, 0/1의 이분형이어야 함
        summary: True로 설정하면 적합된 모델의 요약 통계량을 출력함, Defaults to False
    
    Returns:
        적합이 완료된 로지스틱 회귀분석 결과 객체
    """
    x=data.drop(columns=[y])   # 독립변수 데이터프레임 생성
    y_series = data[y]         # 종속변수 시리즈 생성
    x_input = add_constant(x)  # 독립변수에 절편(상수항) 추가
    model = Logit(y_series, x_input) # Logit 모델 객체 생성
    fit = model.fit(disp=0)    # 모델 적합. disp=0 -> 수렴 메시지 출력 안함

    # 적합된 모델의 요약 통계량 출력 여부 확인
    if summary:
        print(fit.summary())

    # 적합된 모델 객체(분석 결과) 반환
    return fit
    

#--------------------------------
# 예측값 생성 함수 정의
#--------------------------------
def predict(fit, new_data, threshold=0.5):
    """
    적합된 로지스틱 모델로 새로운 데이터의 예측 확률과 예측 범주를 계산한다.

    로지스틱 회귀의 예측값은 '1'(사건 발생)일 확률이므로, 임계값(threshold)을 초과하면 1, 그렇지 않으면 0으로 분류한다.

    Args:
        fit: 'fit_model' 함수로 적합된 회귀분석 결과 객체
        new_data: 예측에 사용할 새로운 데이터프레임. 독립변수 컬럼만 포함해야 한다.
        threshold (float): 확률을 0/1로 분류하는 임계값 (기본값:0.5)

    Returns:
        DataFrame : 예측확률('proba')과 예측값('pred')을 담은 데이터프레임
    """
    # 새로운 데이터에 절편(상수항) 추가
    new_data_with_const = add_constant(new_data)

    # 사건 발생(=1) 확률 예측
    proba = fit.predict(new_data_with_const)

    # 예측 확률과 임계값 기준 예측 범주를 DataFrame으로 반환
    return DataFrame({
        'proba':proba,                          # 1이 될 확률
        'pred':(proba > threshold).astype(int)  # 예측값
    })



#--------------------------------
# 시그모이드 곡선 시각화 함수 정의
#--------------------------------

def plot_sigmoid(fit, data, x, threshold=0.5, palette=None, title=None, xlabel=None, ylabel=None, 
                 width=1280, height=640, save_path=None):
    """
    독립변수에 따른 사건 발생 확률의 S자 곡선(시그모이드)을 그린다.

    Args:
        fit : 'fit_model' 함수로 적합된 회귀분석 결과 객체
        data : 회귀분석에 사용한 원본 데이터프레임(독립변수와 종속변수를 모두 포함)
        x (str): 곡선의 x축으로 사용할 독립변수명
        threshold (float) : 확률을 0/1로 분류하는 임계값 (기본값: 0.5)
        palette (str) : 그래프 색상에 사용할 팔레트 이름(기본값 : None)
        title (str) : 그래프 제목 (기본값 : None)
        xlabel (str) : x축 레이블 (기본값 : None -> 독립변수 명)
        ylabel (str) : y축 레이블 (기본값 : None -> 'P(종속변수=1)')
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 캔버스 세로 픽셀 (기본값: 640).
        save_path (str): 이미지 저장 경로 (기본값: None).
    """
        # --- 1) 그림 종속변수 결정 ---
    yname = fit.model.endog_names  # 종속변수 이름은 모델 객체에서 추출 가능

    # --- 2) 곡선을 그릴 x값 격자 생성 ---
    # 관측된 x의 최소값~최댓값을 200등분해 촘촘한 곡선을 만든다
    grid = np.linspace(data[x].min(), data[x].max(), 200)

    # 곡선 계산용 입력 데이터
    curve_input = DataFrame({x: grid})

    # 예측에 사용할 수 있도록 상수항을 추가한 뒤 사건 발생(=1) 확률 계산
    proba = fit.predict(add_constant(curve_input))

    # --- 3) 그래프 초기화 ---
    line_color = sb.color_palette(palette)[0] if palette else "#328C"

    fig, ax = my_plot.init(width=width, height=height, title=title,
                           xlabel=xlabel if xlabel else x,
                           ylabel=ylabel if ylabel else f"P({yname}=1)")

    # --- 4) 실제 관측치(0/1) 산점도 ---
    # 같은 높이(0 또는 1)에 점이 겹쳐 보이므로 반투명하게 처리한다
    my_plot.scatterplot(data=data, x=x, y=yname, color="#888888",
                        alpha=0.4, palette=None, ax=ax)

    # --- 5) 예측 확률의 S자 곡선 ---
    my_plot.lineplot(x=grid, y=proba, color=line_color, ax=ax)

    # --- 6) 임계값 가로선과 분류 경계 세로선 ---
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.7)

    # 곡선이 임계값을 넘어서는 지점(= 분류 경계)을 찾아 세로선으로 표시한다.
    # 확률이 단조 증가/감소하므로 부호가 바뀌는 첫 지점을 경계로 본다.
    crossed = np.where(np.diff(np.sign(np.asarray(proba) - threshold)) != 0)[0]
    if crossed.size > 0:
        boundary = grid[crossed[0]]
        ax.axvline(x=boundary, color="red", linestyle="--", alpha=0.7)
        ax.text(x=boundary, y=threshold, s=f" 분류 경계: {boundary:.2f}",
                color="red", va="bottom", ha="left")

    # 확률은 0~1 범위이므로 여백을 조금 두고 축을 고정한다
    ax.set_ylim(-0.1, 1.1)

    # --- 7) 그래프 표시 ---
    my_plot.show(save_path=save_path)


#--------------------------------
# 모델 적합도 보고 함수 정의
#--------------------------------
def report_fitness(fit):
    """
    적합된 로지스틱 모델의 모형 적합도(model fit) 보고 문장을 생성해 반환한다.

    summary() 결과표의 문자열을 파싱하지 않고, 'fit' 객체가 이미 갖고 있는 속성에서 지표를 직접 읽어와 문장을 구성한다.

    Args:
        fit: 'fit_model' 함수로 적합된 회귀분석 결과 객체

    Returns:
        str: 모형 적합도 보고 문장. 'Ipython.display.Markdown'으로 감싸 출력하면 좋다.
    """
    # --- 1) 변수 라벨 구성 (상수항 제외) ---
    yname = fit.model.endog_names
    xnames = [name for name in fit.model.exog_names if name != 'const']
    xlabel = ','.join(xnames)

    # --- 2) 유의확률 구간 표기 변환 ---
    p = fit.llr_pvalue
    if p < 0.001: alpha = "< 0.001"
    elif p < 0.01: alpha = "< 0.01"
    elif p < 0.05: alpha = "< 0.05"
    else:          alpha = "≥ 0.05"

    # 유의수준(0.05) 기준 모형의 통계적 유의성 판정
    result = "유의하였다" if p < 0.05 else "유의하지 않았다"

    # --- 3) 유사결정계수의 적합 수준 해석 ---
    # (fit.presquared 관례 : 0.2~0.4 매우 우수/ 0.1~0.2 양호 / 그 미만 다소 낮음)
    prsq = fit.prsquared
    if prsq >= 0.2:     fit_level = "매우 우수한"
    elif prsq >= 0.1:   fit_level = "양호한"
    else:               fit_level = "다소 낮은"
    

    # --- 4) 문장 템플릿 구성 ---
    template = (
        "**Note. n = {n}. "
        "LL = {llf}, LL-Null = {llnull}, "
        "LLR X²({df_model}) = {llr}, p {alpha}, "
        "Pseudo R² = {prsq}**\n\n"
        "{Y}를 종속변수로, {X}(을)를 독립변수로 한 이항 로지스틱 회귀분석 결과, "
        "모형은 통계적으로 {result}.\n\n"
        "> LLR X²({df_model}) = {llr}, p {alpha} , Pseudo R² = {prsq}.\n\n"
        "즉, 모형의 유사결정계수는 {prsq}로 {fit_level} 적합수준을 보였다.\n\n"
        "> ※Pseudo R²는 선형회귀의 R²처럼 '분산 설명 비율'로 해석하지 않는다. "
        "일반적으로 **0.2~0.4** 구간이면 매우 우수한 적합으로 본다.")

    # --- 5) 문장 템플릿 값 치환 ---
    report =template.format(
        n=int(fit.nobs),
        llf=round(fit.llf, 3),
        llnull=round(fit.llnull, 3),
        df_model=int(fit.df_model),
        llr=round(fit.llr, 3),
        alpha=alpha,
        prsq=round(prsq, 3),
        Y=yname,
        X=xlabel,
        result=result,
        fit_level=fit_level
    )

    # --- 6) 결과 리턴 ---
    return report


#--------------------------------
# 독립변수 보고표 생성 함수 정의
#--------------------------------
def report_variables(fit, data):
    """
    적합된 로지스틱 모델의 독립변수별 회귀계수, 오즈비 보고표를 데이터프레임으로 생성해 반환한다.

    오즈비와 그 95% 신뢰구간을 함께 제공한다.
    다중공선성 점검을 위한 VIF 계산에 원본 데이터가 필요하므로 'data'를 함께 받는다.

    Args:
        fit: 'fit_model'함수로 적합된 회귀분석 결과 객체
        data: 회귀분석에 사용한 원본 데이터프레임. 독립변수와 종속변수를 모두 포함해야한다.
        
    Returns:
        DataFrame: 독립변수별 보고표. 종속변수, 독립변수, B, 표준오차, z, 유의확률, 오즈비(OR), OR 95% 신뢰구간, 공차, VIF 컬럼을 가진다.
                |B|(=|log OR|) 내림차순으로 정렬되어 영향력이 큰 변수가 위로 온다.
    """
    # --- 1) 대상 변수 확인 및 VIF 계산 ---
    yname = fit.model.endog_names               # 종속변수 이름
    exog_names = list(fit.model.exog_names)     # 상수항(const)을 포함한 전체 변수 이름 순서
    xnames = [name for name in exog_names if name != 'const'] # 상수항 제외 독립변수

    vif = my_stats.compute_vif(data, columns=xnames)

    # 통계량을 위치 인덱스로 접근하기 위해 배열로 전환
    params = np.asarray(fit.params)            # 비표준화 회귀계수(B)
    bse = np.asarray(fit.bse)                  # 계수 표준오차
    zvalues = np.asarray(fit.tvalues)          # z-통계량 (로지스틱은 t가 아니라 z)
    pvalues = np.asarray(fit.pvalues)          # 계수 유의확률
    conf = np.asarray(fit.conf_int())          # 계수의 95% 신뢰구간 [하한, 상한]

    # --- 2) 독립변수별 계수, 오즈비 정리 ---
    variables =[]  # 독립변수별 보고 내용을 저장할 빈 리스트
    for x in xnames:
        i = exog_names.index(x)             # 상수항을 포함한 전체 순서에서의 위치
        b = float(params[i])                # 비표준화 회귀계수(B)
        vif_value = vif.loc[x, "VIF"]       # 미리 계산해 둔 VIF 값 조회

        # 표준화 회귀계수
        # --> (βstd) = B x 독립변수 표준편차 = "독립변수 1 SD 변화당 log(오즈비) 변화".
        # 로지스틱은 종속변수가 0/1이라 (OLS와 달리) y로는 표준화하지 않고 독립변수만 표준화한다.
        beta_std = b * float(data[x].std(ddof=1))
        row = {
            "종속변수" : yname,                         # 종속변수 이름                                              
            "독립변수" : x,                             # 독립변수 이름                                        
            "B": b,                                    # 비표준화 회귀계수(B)     
            "βstd" : beta_std,                         # 표준화 회귀계수                                      
            "표준오차": bse[i],                         # 계수 표준오차                               
            "z": zvalues[i],                           # z - 통계량                                
            "유의확률": pvalues[i],                     # 계수 유의확률                    
            "오즈비(OR)":float(np.exp(b)),                  # 오즈비 = exp(B)
            "OR 95% 하한" : float(np.exp(conf[i,0])),   # 오즈비 신뢰구간 하한
            "OR 95% 상한" : float(np.exp(conf[i,1])),   # 오즈비 신뢰구간 상한                                              
            "공차": 1/ vif_value,                       # 공차(Tolerance = 1/VIF)                              
            "VIF": vif_value                            # 분산팽창계수
            }
        variables.append(row)
    

    # --- 3) 독립변수별 보고표 생성 및 반환 ---
    vdf = DataFrame(variables)

    # |B|(=|log OR|)의 절대값으로 내림차순 정렬 후 리턴 (영향력이 큰 변수가 위로 오도록)
    vdf = vdf.sort_values("B", key=abs, ascending=False).reset_index(drop=True)
    return vdf


#--------------------------------
# 오즈비 해석 문장 생성 함수 정의
#--------------------------------
def report_variables_text(fit, data=None, alpha=0.05):
    """
    독립변수별 오즈비 해석 문장을 markdown 불릿 리스트로 생성해 반환한다.

    각 독립변수에 대해 계수(B), 오즈비(OR), z, 유의확률을 문장으로 풀어쓰고,
    오즈비를 백분율 ㅂ녀화로 환산하여 "오즈가 약 몇 % 증가/감소" 형태로 해석한다.
    'data'가 주어지면 이분형(더미) 변수와 연속형 변수를 구분해 해석 표현을 달리한다.

    Args:
        fit: 'fit_model'함수로 적합된 회귀분석 결과 객체
        data: 원본 데이터프레임(기본값 : None). 주어지면 이분형/연속형을 구분해 표현한다.
        alpha (float): 유의성 판정에 사용할 유의수준 (기본값 : 0.05)

    Returns:
        str: 독립변수별 해석 문장 불릿 리스트. 'Ipython.display.Markdown'으로 감싸 출력하면 좋다.
    """
    # --- 1) 해석 대상 결정 (상수항 제외) ---
    yname = fit.model.endog_names
    xnames = [name for name in fit.model.exog_names if name != 'const']

    # --- 2) 문장 템플릿 구성 (독립변수마다 반복 적용) ---
    line_template = (
        "- **{x}**의 회귀계수는 **B = {B}**, 오즈비는 **OR = {OR}**로 나타났으며, "
        "이는 **{y}**에 {sig} 요인임을 의미한다. "
        "(**z = {z}**, **{p}**) "
        "즉, {change} {y}가 1(사건 발생)이 될 오즈는 평균적으로 약 **{pct}% {direction}**하는 것으로 해석된다. "
    )

    # --- 3) 독립변수별 해석 문장 생성 ---
    lines = []   # 독립변수별 문장(불릿)을 저장할 빈 리스트
    for x in xnames:
        B = fit.params[x]   # 비표준화 회귀계수(B)
        z = fit.tvalues[x]  # z-통계량
        p = fit.pvalues[x]  # 계수 유의확률
        OR = np.exp(B)      # 오즈비 = exp(B)

        # 유의성 판정 (유의수준 기준)
        sig_word = "유의한" if p < alpha else "유의하지 않은"

        # p값 APA 표기(앞자기 0 생략)
        if p < 0.001:  p_text="p < .001"
        else:          p_text=f"p = {p:.3f}".replace("0.", ".")

        # 오즈비를 백분율 변화로 환산 (OR>1 증가, OR<1 감소)
        pct = abs((OR -1)*100)
        direction = "증가" if B > 0 else "감소"

        # 변화 표현 : 이분형(더미)이면 '기준 범주 대비 해당 범주', 연속형이면 '1 증가'
        is_binary = data is not None and data[x].nunique() <=2
        if is_binary:       change = f"**{x}**에 해당하는 경우(기준 범주 대비)"
        else:               change = f"**{x}**가 1 증가할 때"

        # 하나의 독립변수 -> 하나의 불릿 문장
        lines.append(line_template.format(
            x=x, B=round(B, 4), OR=round(OR, 4), y=yname, sig=sig_word,
            z=round(z, 2), p=p_text, change=change,
            pct=round(pct, 1), direction=direction
        ))

    # --- 4) 해석 주의 각주 첨부 ---
    report = "\n".join(lines)
    report += (
        "\n\n> ※ 오즈비(OR)가 1보다 크면 사건 발생 오즈가 증가, 1보다 작으면 감소함을 뜻한다. "
        "유의확률이 유의수준보다 큰(=유의하지 않은) 변수는 효과가 통계적으로 확인되지 않았으므로 "
        "오즈비 해석에 주의한다. 더미변수의 오즈비는 '기준(drop_first로 제외된) 범주' 대비 값이다. "
    )

    return report


#--------------------------------
# 오즈비 시각화 함수 정의
#--------------------------------
def plot_odds(fit, data, palette=None, title=None, xlabel=None, ylabel=None,
              width=1280, height=None, save_path=None):

    """
    오즈비(Odds Ratio)를 가로 막대그래프로 시각화해 독립변수의 영향력을 보여준다.

    막대는 `report_variables`가 정렬해 둔 |B|(=|log OR|) 내림차순 그대로 위에서
    아래로 배치되며, 오즈비가 1보다 큰지(사건 발생 오즈 증가) 작은지(감소) 에 따라
    색을 달리하고 막대 끝에 오즈비 값을 표기한다. OR=1(영향 없음) 위치에 기준선을 둔다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        palette (dict): 부호별 막대 색상. None이면 {'+': 파랑, '-': 빨강} (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None → "오즈비(Odds Ratio)").
        ylabel (str): y축 레이블 (기본값: None → "독립변수").
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 캔버스 세로 픽셀. None이면 독립변수 수 × 80으로 자동 계산 (기본값: None).
        save_path (str): 이미지 저장 경로 (기본값: None).
    """
    # --- 1) 시각화용 데이터 전처리 ---
    vdf = report_variables(fit, data)
    rdf = vdf[["독립변수", "오즈비(OR)"]].copy()
    
    # OR>1이면 증가(+), OR<1이면 감소(-)로 색상 구분
    rdf["부호"] = np.where(rdf['오즈비(OR)'] > 1, "+", "-")

    # 독립변수가 많을수록 막대가 촘촘해지므로, 변수 하나당 80px씩 세로 공간을 확보한다.
    if height is None:
        height = len(rdf) * 80

    if height < 200:
        height = 200 # 최소 높이 200px

    # 부호별 기본 색상: 증가(+)은 파랑, 감소(-)은 빨강
    if palette is None:
        palette = {"+": "#0066ff", "-":"#ff3333"}

    # --- 2) 그래프 초기화 ---
    fig, ax = my_plot.init(width=width, height=height, title=title, xlabel=xlabel if xlabel else "오즈비(Odds Ratio)",
                           ylabel=ylabel if ylabel else "독립변수")

    # --- 3) 가로 막대그래프 ---
    my_plot.barplot(rdf, x="오즈비(OR)", y="독립변수", hue='부호', palette=palette, ax=ax)

    # OR=1(영향 없음) 기준선
    ax.axvline(x=1, color='gray', linestyle="--", alpha=0.7)

    # --- 4) 막대 끝에 오즈비 값 표기 ---
    for i in rdf.index:
        orv = rdf.loc[i, "오즈비(OR)"]
        ax.text(x=orv, y=i, s=f"{orv:.2f}", va='center', ha='left' if orv >= 1 else 'right', color = 'black')

    # --- 5) 그래프 표시 ---
    my_plot.show(save_path=save_path)


def auto_logit(data, y, report=True, plot=True, threshold=0.5,
               width=1280, height=640, backward=False, alpha=0.05):
    """로지스틱 회귀모델 적합부터 변수 선택·보고서 출력·시각화까지 한 번에 수행한다.

    `backward=True`이면 유의하지 않은 독립변수가 모두 사라질 때까지
    후진소거법(유의확률이 가장 큰 변수를 하나씩 제거 후 재적합)을 반복한다.

    Args:
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        y: 종속변수로 사용할 컬럼명 (0/1 이분형).
        report (bool): 오즈비 보고표·모형 적합도·해석 문장 출력 여부 (기본값: True).
        plot (bool): 시각화 출력 여부 (기본값: True).
        threshold (float): 시그모이드 곡선에 표시할 분류 임계값 (기본값: 0.5).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
        backward (bool): 후진소거법으로 유의하지 않은 독립변수를 제거할지 여부 (기본값: False).
        alpha (float): 후진소거법의 변수 제거 기준이자 해석 문장의 유의수준 (기본값: 0.05).

    Returns:
        적합이 완료된 로지스틱 회귀분석 결과 객체.
    """
    # 빈 줄 출력 (출력 결과의 여백을 위함)
    print()

    # --- 1) 모델 적합 (backward=True이면 유의하지 않은 변수가 없어질 때까지 반복) ---
    while True:
        fit = fit_model(data, y)

        if not backward:
            break  # 후진소거법이 아니면 반복문 종료

        pvalues = report_variables(fit, data)["유의확률"]

        # 종료 조건: 독립변수가 하나뿐이거나 남은 변수가 모두 유의한 경우
        if len(pvalues) <= 1 or pvalues.max() < alpha:
            break

        # 유의확률이 가장 큰(=가장 유의하지 않은) 독립변수를 하나만 제거한다.
        # 여러 개를 한꺼번에 지우면, 변수 간 상관 때문에 원래는 유의해졌을 변수까지 사라진다.
        worst = report_variables(fit, data).loc[pvalues.idxmax(), "독립변수"]
        print(f"유의하지 않은 독립변수 제거 → {worst} (p = {pvalues.max():.4f})")
        data = data.drop(columns=[worst])

    # --- 2) 모형 적합도 및 독립변수 보고 ---
    if report:
        display(Markdown("#### ▶ 모형 적합도"))
        display(report_variables(fit, data))                       # 오즈비 보고 표
        display(Markdown(report_fitness(fit)))                     # 모형 적합도 해설
        display(Markdown(report_variables_text(fit, data=data, alpha=alpha)))  # 변수별 해석

    # --- 2) 시각화 ---
    if plot:
        display(Markdown("#### ▶ 오즈비 시각화"))
        plot_odds(fit, data, width=width)

    # --- 4) 최종 적합 모델 객체 반환 ---
    return fit

def plot_confusion(fit, threshold=0.5, palette=None, title=None,
                   size=640, save_path=None, ax=None):
    """
    혼동행렬(Confusion Matrix)을 정사각형 히트맵으로 시각화한다.

    Args:
        fit: 'fit_model' 함수로 적합된 회귀분석 결과 객체
        threshold (float): 확률을 0/1로 분류하는 임계값 (기본값:0.5)
        palette (str) : 히트맵 색상 팔레트 이름 (기본값: None → "Blues")
        title (str) : 그래프 제목 (기본값: None → "혼동행렬(Confusion Matrix)")
        size (int): 캔버스 한 변의 픽셀 크기. 가로, 세로가 같아(기본값 : 640)
        save_path (str): 이미지 저장 경로 (기본값: None)
        ax: 그래프를 그릴 Axes 객체 (기본값: None → 캔버스를 새로 생성)
    """
    # --- 1) 실제값과 임계값 기준 예측 범주로 혼동행렬 구성 ---
    y_true = np.asarray(fit.model.endog).astype(int)
    y_pred = (np.asarray(fit.predict()) > threshold).astype(int)
    cm =confusion_matrix(y_true, y_pred)

    cmdf = DataFrame(cm, 
                     index=["실제 0 (Negative)", "실제 1 (Positive)"],
                     columns=["예측 0 (Negative)", "예측 1 (Positive)"])

    # --- 2) 시각화 ---
    # ax를 넘겨받으면 서브플롯의 한칸에 그리고, 없으면 단독 캔버스를 만든다.
    fig = None
    if ax is None:
        fig, ax = my_plot.init(width=size, height=size)

    ax.set_title(title if title else "혼동행렬(Confusion Matrix)",
                 fontsize=24, fontweight=500, pad=15)

    # square=True로 셀을 정사각형으로 맞추고, 값이 표기되므로 색상막대(cbar)는 생략한다
    # 빈도(정수)를 그대로 표기하도록 fmt="d" 사용
    my_plot.heatmap(data=cmdf, annot=True, fmt="d",
                    palette=palette if palette else 'Blues',
                    annot_kws={'size': size*0.07}, # 글자크기를 그래프 크기의 7%로 설정
                    square=True, cbar=False, ax=ax)

    if fig is not None:
        my_plot.show(save_path=save_path)

def plot_roc(fit, palette=None, title=None, size=640, save_path=None, ax=None):
    """ROC Curve를 정사각형으로 그리고 AUC(곡선 아래 면적)를 제목에 표시한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        palette (str): 곡선 색상에 사용할 팔레트 이름 (기본값: None → 파랑 단색).
        title (str): 그래프 제목 (기본값: None → "ROC Curve (AUC = ...)").
        size (int): 캔버스 한 변의 픽셀 크기. 가로·세로가 같다 (기본값: 640).
        save_path (str): 이미지 저장 경로 (기본값: None).
        ax: 그래프를 그릴 Axes 객체 (기본값: None → 캔버스를 새로 생성).
    """
    # --- 1) 예측 확률로 ROC 좌표와 AUC 계산 ---
    y_true = np.asarray(fit.model.endog).astype(int)    # 실제 종속변수 값
    proba = np.asarray(fit.predict())                   # 모델의 예측값 --> 1이 될 확률
    auc = roc_auc_score(y_true, proba)                  # AUC 계산
    roc_fpr, roc_tpr, _ = roc_curve(y_true, proba)      # ROC 좌표 (FPR,TPR,임계값)

    # --- 2) 시각화 초기화 ---
    # 팔레트가 지정되면 첫 번째 색을 곡선 색상으로 사용
    line_color = sb.color_palette(palette)[0] if palette else "#328CC1"

    fig = None
    if ax is None:
        fig, ax = my_plot.init(width=size, height=size)

    # x축=위양성률(FPR), y축=재현율(TPR)  ← ROC의 표준 축 배치
    ax.set_title(title if title else f"ROC Curve (AUC = {auc:.4f})",
                 fontsize=24, fontweight=500, pad=15)
    ax.set_xlabel("위양성률(FPR, 1 - 특이성)", fontsize=16, fontweight=400, labelpad=5)
    ax.set_ylabel("재현율(TPR, 민감도)", fontsize=16, fontweight=400, labelpad=5)

    # --- 3) ROC 곡선과 기준선 그리기 ---
    my_plot.lineplot(x=roc_fpr, y=roc_tpr, color=line_color, ax=ax)

    # 무작위로 찍었을 때의 기준선(대각선)
    my_plot.lineplot(x=[0, 1], y=[0, 1], color="red", linestyle="--", ax=ax)

    # x·y 모두 0~1 범위이므로 축 비율을 1:1로 맞추면 그래프 영역이 정사각형이 된다
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    if fig is not None:
        my_plot.show(save_path=save_path)


def report_performance(fit, threshold=0.5, plot=True, palette=None,
                       size=640, save_path=None):
    """적합된 로지스틱 모델의 분류 성능을 혼동행렬과 평가지표로 정리해 출력한다.

    위양성률(FPR)과 특이성(TNR)은 sklearn에 함수가 없어 혼동행렬로 직접 계산한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        threshold (float): 확률을 0/1로 분류하는 임계값 (기본값: 0.5).
        plot (bool): 혼동행렬 히트맵과 ROC Curve를 함께 그릴지 여부 (기본값: True).
        palette (str): 그래프 색상에 사용할 팔레트 이름 (기본값: None).
        size (int): 서브플롯 한 칸의 한 변 픽셀 크기 (기본값: 640).
        save_path (str): 이미지 저장 경로 (기본값: None).
    """
    # --- 1) 실제값·예측확률·예측범주 준비 ---
    y_true = np.asarray(fit.model.endog).astype(int)    # 실제 종속변수(0/1)
    proba = np.asarray(fit.predict())                   # 1이 될 확률
    y_pred = (proba > threshold).astype(int)            # 임계값 기준 예측 범주(0/1)

    # --- 2) 혼동행렬 및 TN/FP/FN/TP 분해 ---
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # --- 3) 평가지표 계산 ---
    fallout = fp / (fp + tn) if (fp + tn) > 0 else 0.0       # 위양성률(FPR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # 특이성(TNR = 1 - FPR)

    auc = roc_auc_score(y_true, proba)

    # AUC 구간별 판정 (0.5는 동전 던지기와 같아 학습이 되지 않은 상태)
    if auc < 0.6:
        auc_grade = "모형이 아무 것도 학습하지 못함"
    elif auc < 0.7:
        auc_grade = "낮음"
    elif auc < 0.8:
        auc_grade = "쓸 만함"
    elif auc < 0.9:
        auc_grade = "우수"
    else:
        auc_grade = "매우 우수"

    # 진단오즈비 = (TP×TN)/(FP×FN). 오분류가 하나도 없으면 분모가 0이므로 무한대로 둔다
    if fp * fn > 0:
        dor = (tp * tn) / (fp * fn)
    else:
        dor = np.inf

    # --- 4) 평가표 구성 ---
    metrics = DataFrame([{
        "정확도(Accuracy)": accuracy_score(y_true, y_pred),
        "정밀도(Precision)": precision_score(y_true, y_pred, zero_division=0),
        "재현율(Recall,TPR)": recall_score(y_true, y_pred, zero_division=0),
        "위양성율(Fallout,FPR)": fallout,
        "특이성(Specificity,TNR)": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": auc,
        "AUC 판단": auc_grade,
        "진단오즈비(DOR)": dor,
    }], index=["performance"])

    # --- 5) 혼동행렬 ---
    cmdf = DataFrame(cm,
                     index=["실제 0 (Negative)", "실제 1 (Positive)"],
                     columns=["예측 0 (Negative)", "예측 1 (Positive)"])

    # --- 6) 결과 출력 ---
    display(cmdf)
    display(metrics)

    # --- 7) 시각화: 혼동행렬 히트맵 + ROC Curve를 1행 2열로 나란히 배치 ---
    if plot:
        fig, ax = my_plot.init(width=size, height=size, rows=1, cols=2)
        plot_confusion(fit, threshold=threshold, palette=palette, ax=ax[0])
        plot_roc(fit, palette=palette, ax=ax[1])
        my_plot.show(save_path=save_path)


def test_linear(fit, data, alpha=0.05):
    """Box-Tidwell 검정으로 연속형 독립변수의 로짓 선형성을 점검한다.

    독립변수 x에 `x × ln(x)` 항을 하나씩 추가해 그 항이 유의한지 본다.
    유의하면(p < alpha) 로짓이 직선이 아니라는 뜻이므로 변환·제곱항·구간화가 필요하다.
    이분형(더미) 변수는 두 점을 잇는 선이 언제나 직선이므로 검정 대상에서 제외한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 회귀분석에 사용한 원본 데이터프레임. 독립변수와 종속변수를 모두
            포함해야 한다.
        alpha (float): 유의수준 (기본값: 0.05).
    """
    # --- 1) 검정 대상 결정 (상수항 제외, 연속형만) ---
    yname = fit.model.endog_names

    xnames = []
    for name in fit.model.exog_names:
        if name != "const":
            xnames.append(name)

    targets = []
    for x in xnames:
        # 값의 종류가 3가지 이상인 변수만 연속형으로 간주한다
        if data[x].nunique() > 2:
            targets.append(x)

    # --- 2) Box-Tidwell 검정 (변수마다 x*ln(x) 항을 하나씩 추가) ---
    rows = []
    for x in targets:
        bt_data = data[xnames].copy()       # 독립변수만 복사
        base = bt_data[x]                   # 원본 독립변수 값

        # 최소값이 0 이하이면 ln 정의역(양수)을 맞추기 위해 평행이동 필요
        shifted = bool(base.min() <= 0)
        if shifted:
            base = base - base.min() + 1

        # Box-Tidwell 항 생성: x × ln(x)
        bt_data["_bt_term"] = base * np.log(base)

        # Logit 모델 적합 후 Box-Tidwell 항의 z·p값으로 선형성 판정
        bt_fit = Logit(data[yname], add_constant(bt_data)).fit(disp=0)
        z = float(bt_fit.tvalues["_bt_term"])
        p = float(bt_fit.pvalues["_bt_term"])
        linearity = bool(p >= alpha)

        # 선형성 판정에 따라 결론 문구를 달리한다
        if linearity:
            conclusion = "귀무가설 채택 → 로짓 선형성 위배 근거 없음"
        else:
            conclusion = "대립가설 채택 → 로짓 선형성 위배(변환 필요)"

        # 독립변수별 결과를 딕셔너리로 정리해 리스트에 추가
        rows.append({
            "z": round(z, 4),
            "p-value": round(p, 4),
            "linearity": linearity,
            "위치이동": shifted,
            "result": conclusion,
            "독립변수": x,
        })

    # 결과표 생성 및 출력
    result_df = DataFrame(rows).set_index("독립변수")
    display(result_df)   # 결과표 출력
