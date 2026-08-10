import numpy as np
import seaborn as sb
from IPython.display import display, Markdown
from pandas import DataFrame, Series, concat, qcut
from statsmodels.api import add_constant, Logit
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import chi2
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, recall_score, precision_score, f1_score,
)

from . import my_plot
from . import my_prep
from . import my_stats


def fit_model(data, y, summary=False):
    """statsmodels의 Logit을 이용해 로지스틱 회귀 모델을 적합한다.

    종속변수 `y`를 제외한 나머지 모든 컬럼을 독립변수로 사용하며,
    절편(상수항)을 자동으로 추가한 뒤 최대우도추정(MLE)으로 회귀계수를 추정한다.
    종속변수는 0/1의 두 값만 가지는 이분형이어야 한다.

    Args:
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        y: 종속변수로 사용할 컬럼명. `data`에 반드시 존재해야 하며 0/1의 이분형이어야 한다.
        summary: True로 설정하면 적합된 모델의 요약 통계량을 출력한다.
                  Defaults to False.

    Returns:
        적합이 완료된 로지스틱 회귀분석 결과 객체.
    """
    x = data.drop(columns=[y])          # 독립변수 데이터프레임 생성
    y_series = data[y]                  # 종속변수 시리즈 생성
    x_input = add_constant(x)           # 독립변수에 절편(상수항) 추가
    model = Logit(y_series, x_input)    # Logit 모델 객체 생성
    fit = model.fit(disp=0)             # 모델 적합. disp=0 -> 수렴 메시지 출력 안함

    if summary:
        print(fit.summary())            # 적합된 모델의 요약 통계량 출력 여부 확인

    return fit                          # 적합된 모델 객체(분석 결과) 반환


def predict(fit, new_data, threshold=0.5):
    """적합된 로지스틱 모델로 새로운 데이터의 예측 확률과 예측 범주를 계산한다.

    다중 로지스틱 회귀에서도 안전하게 동작하도록, 모델이 학습한 독립변수만
    **학습 당시의 순서 그대로** 골라 사용한다. 필요 없는 컬럼이 섞여 있으면 무시하고,
    학습에 사용된 독립변수가 빠져 있으면 어떤 변수가 없는지 알려 준다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        new_data: 예측에 사용할 새로운 데이터. 데이터프레임, 한 건짜리 Series 또는 dict.
        threshold (float): 확률을 0/1로 분류하는 임계값 (기본값: 0.5).

    Returns:
        DataFrame: 예측 확률('proba')과 예측값('pred')을 담은 데이터프레임.

    Raises:
        ValueError: 모델이 학습한 독립변수가 `new_data`에 없는 경우.
    """
    # --- 1) 한 건만 예측하는 경우도 데이터프레임으로 통일 ---
    if isinstance(new_data, dict):
        new_data = DataFrame([new_data])
    elif isinstance(new_data, Series):
        new_data = new_data.to_frame().T

    # --- 2) 모델이 학습한 독립변수 확인 (상수항 제외, 학습 당시의 순서 유지) ---
    xnames = [name for name in fit.model.exog_names if name != "const"]

    # 학습에 사용된 독립변수가 빠져 있으면 예측 자체가 불가능하다
    missing = [name for name in xnames if name not in new_data.columns]
    if missing:
        raise ValueError(
            f"모델이 학습한 독립변수가 new_data에 없습니다: {missing}\n"
            f"필요한 독립변수: {xnames}"
        )

    # 학습 때와 동일한 순서로 컬럼을 맞춘다. (모델에 없는 컬럼은 자동으로 제외)
    # 컬럼 순서가 어긋나면 오류 없이 '조용히 틀린 확률'이 나오므로 반드시 정렬해야 한다.
    x = new_data[xnames].astype(float)

    # --- 3) 절편(상수항) 추가 ---
    # has_constant="add" -> 값이 일정한 컬럼이 있어도 상수항을 빠뜨리지 않는다
    x_input = add_constant(x, has_constant="add")

    # --- 4) 사건 발생(=1) 확률 예측 ---
    proba = fit.predict(x_input)

    # 예측 확률과 임계값 기준 예측 범주를 DataFrame으로 반환
    return DataFrame({
        "proba": proba,                             # 1이 될 확률
        "pred": (proba > threshold).astype(int),    # 예측값
    })


def plot_sigmoid(fit, data, x, threshold=0.5, palette=None, title=None,
                 xlabel=None, ylabel=None, width=1280, height=640, save_path=None):
    """독립변수에 따른 사건 발생 확률의 S자 곡선(시그모이드)을 그린다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 회귀분석에 사용한 원본 데이터프레임. (독립변수와 종속변수를 모두 포함)
        x (str): 곡선의 x축으로 사용할 독립변수명.
        threshold (float): 확률을 0/1로 분류하는 임계값 (기본값: 0.5).
        palette (str): 그래프 색상에 사용할 팔레트 이름. (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None → 독립변수명).
        ylabel (str): y축 레이블 (기본값: None → "P(종속변수=1)").
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 캔버스 세로 픽셀 (기본값: 640).
        save_path (str): 이미지 저장 경로 (기본값: None).
    """
    # --- 1) 그릴 종속변수 결정 ---
    yname = fit.model.endog_names

    # --- 2) 곡선을 그릴 x값 격자 생성 ---
    # 관측된 x의 최솟값~최댓값을 200등분해 촘촘한 곡선을 만든다
    grid = np.linspace(data[x].min(), data[x].max(), 200)

    # 곡선 계산용 입력 데이터
    curve_input = DataFrame({x: grid})

    # 예측에 사용할 수 있도록 상수항을 추가한 뒤 사건 발생(=1) 확률 계산
    proba = fit.predict(add_constant(curve_input))

    # --- 3) 그래프 초기화 ---
    # 팔레트가 지정되었다면 첫 번째 색상을 선 색상으로 사용하고, 지정되지 않았다면 기본 파랑색을 사용한다
    line_color = sb.color_palette(palette)[0] if palette else "#328CC1"

    # 그래프 초기화 및 축 레이블 설정
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
    a = fit.params[x]           # 독립변수 x에 대한 기울기
    b = fit.params['const']     # 절편
    boundary = -b / a           # 분류 경계 = -절편/기울기

    # 분류 경계에 대한 세로 선과 텍스트
    ax.axvline(x=boundary, color="red", linestyle="--", alpha=0.7)
    ax.text(x=boundary, y=threshold, s=f" 분류 경계: {boundary:.2f}",
        color="red", va="bottom", ha="left")

    # 임계값에 대한 가로 선
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.7)
        
    # 확률은 0~1 범위이므로 여백을 조금 두고 축을 고정한다
    ax.set_ylim(-0.1, 1.1)

    # --- 7) 그래프 표시 ---
    my_plot.show(save_path=save_path)


def report_fitness(fit):
    """적합된 로지스틱 모델의 모형 적합도(model fit) 보고 문장을 생성해 반환한다.

    summary() 결과표의 문자열을 파싱하지 않고, `fit` 객체가 이미 갖고 있는 속성에서
    지표를 직접 읽어와 문장을 구성한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.

    Returns:
        str: 모형 적합도 보고 문장. `IPython.display.Markdown`으로 감싸 출력하면 좋다.
    """
    # --- 1) 변수 라벨 구성 (상수항 제외) ---
    yname = fit.model.endog_names
    xnames = [name for name in fit.model.exog_names if name != "const"]
    xlabel = ", ".join(xnames)

    # --- 2) 유의확률 구간 표기 변환 ---
    p = fit.llr_pvalue
    if p < 0.001:   alpha = "< 0.001"
    elif p < 0.01:  alpha = "< 0.01"
    elif p < 0.05:  alpha = "< 0.05"
    else:           alpha = "≥ 0.05"

    # 유의수준(0.05) 기준 모형의 통계적 유의성 판정
    result = "유의하였다" if p < 0.05 else "유의하지 않았다"

    # --- 3) 유사결정계수의 적합 수준 해석 ---
    # (fit.prsquared 관례: 0.2~0.4 매우 우수 / 0.1~0.2 양호 / 그 미만 다소 낮음)
    prsq = fit.prsquared
    if prsq >= 0.2:     fit_level = "매우 우수한"
    elif prsq >= 0.1:   fit_level = "양호한"
    else:               fit_level = "다소 낮은"

    # --- 4) 문장 템플릿 구성 ---
    template = (
        "**Note. n = {n}. "
        "LL = {llf}, LL-Null = {llnull}, "
        "LLR χ²({df_model}) = {llr}, p {alpha}, "
        "Pseudo R² = {prsq}**\n\n"
        "{Y}를 종속변수로, {X}(을)를 독립변수로 한 로지스틱 회귀분석 결과, "
        "모형은 통계적으로 {result}.\n\n"
        "> LLR χ²({df_model}) = {llr}, p {alpha}, Pseudo R² = {prsq}.\n\n"
        "즉, 모형의 유사결정계수는 {prsq}로 {fit_level} 적합 수준을 보였다.\n\n"
        "> ※ Pseudo R²는 선형회귀의 R²처럼 '분산 설명 비율'로 해석하지 않는다. "
        "일반적으로 **0.2~0.4** 구간이면 매우 우수한 적합으로 본다."
    )

    # --- 5) 문장 템플릿 값 치환 ---
    report = template.format(
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
        fit_level=fit_level,
    )

    # --- 6) 결과 리턴 ---
    return report


def report_variables(fit, data):
    """적합된 로지스틱 모델의 독립변수별 회귀계수·오즈비 보고표를 데이터프레임으로 생성해 반환한다.

    오즈비와 그 95% 신뢰구간을 함께 제공한다.
    다중공선성 점검을 위한 VIF 계산에 원본 데이터가 필요하므로 `data`를 함께 받는다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 회귀분석에 사용한 원본 데이터프레임. 독립변수와 종속변수를 모두 포함해야 한다.

    Returns:
        DataFrame: 독립변수별 보고표. 종속변수·독립변수·B·표준오차·z·유의확률·
            오즈비(OR)·OR 95% 신뢰구간·공차·VIF 컬럼을 가진다.
            |B|(=|log OR|) 내림차순으로 정렬되어 영향력이 큰 변수가 위로 온다.
    """
    # --- 1) 대상 변수 확인 및 VIF 계산 ---
    yname = fit.model.endog_names                # 종속변수 이름
    exog_names = list(fit.model.exog_names)      # 상수항(const)을 포함한 전체 변수 이름 순서
    xnames = [name for name in exog_names if name != "const"]  # 상수항 제외 독립변수

    # 독립변수 전체를 대상으로 VIF를 한 번에 계산 (상수항 제외한 결과가 반환된다)
    vif = my_stats.compute_vif(data, columns=xnames)

    # 통계량을 위치 인덱스로 접근하기 위해 배열로 변환
    params = np.asarray(fit.params)             # 비표준화 회귀계수(B)
    bse = np.asarray(fit.bse)                   # 계수 표준오차
    zvalues = np.asarray(fit.tvalues)           # z-통계량 (로지스틱은 t가 아니라 z)
    pvalues = np.asarray(fit.pvalues)           # 계수 유의확률
    conf = np.asarray(fit.conf_int())           # 계수의 95% 신뢰구간 [하한, 상한]

    # --- 2) 독립변수별 계수·오즈비 정리 ---
    variables = []   # 독립변수별 보고 내용을 저장할 빈 리스트
    for x in xnames:
        i = exog_names.index(x)                 # 상수항을 포함한 전체 순서에서의 위치
        b = float(params[i])                    # 비표준화 회귀계수(B)
        vif_value = vif.loc[x, "VIF"]           # 미리 계산해 둔 VIF 값 조회

        # 표준화 회귀계수
        # --> (βstd) = B × 독립변수 표준편차 = "독립변수 1 SD 변화당 log(오즈비) 변화".
        # 로지스틱은 종속변수가 0/1이라 (OLS와 달리) y로는 표준화하지 않고 독립변수만 표준화한다.
        beta_std = b * float(data[x].std(ddof=1))

        row = {
            "종속변수": yname,                          # 종속변수 이름
            "독립변수": x,                              # 독립변수 이름
            "B": b,                                    # 비표준화 회귀계수(B)
            "βstd": beta_std,                          # 표준화 회귀계수
            "표준오차": bse[i],                         # 계수 표준오차
            "z": zvalues[i],                           # z-통계량
            "유의확률": pvalues[i],                     # 계수 유의확률
            "오즈비(OR)": float(np.exp(b)),             # 오즈비 = exp(B)
            "OR 95% 하한": float(np.exp(conf[i, 0])),  # 오즈비 신뢰구간 하한
            "OR 95% 상한": float(np.exp(conf[i, 1])),  # 오즈비 신뢰구간 상한
            "공차": 1 / vif_value,                     # 공차(Tolerance = 1/VIF)
            "VIF": vif_value,                          # 분산팽창계수
        }
        variables.append(row)

    # --- 3) 독립변수별 보고표 생성 및 반환 ---
    vdf = DataFrame(variables)

    # |B|(=|log OR|)의 절대값으로 내림차순 정렬 후 리턴 (영향력이 큰 변수가 위로 오도록)
    vdf = vdf.sort_values("B", key=abs, ascending=False).reset_index(drop=True)
    return vdf


def report_variables_text(fit, data=None, alpha=0.05):
    """독립변수별 오즈비 해석 문장을 markdown 불릿 리스트로 생성해 반환한다.

    각 독립변수에 대해 계수(B)·오즈비(OR)·z·유의확률을 문장으로 풀어 쓰고,
    오즈비를 백분율 변화로 환산하여 "오즈가 약 몇 % 증가/감소" 형태로 해석한다.
    `data`가 주어지면 이분형(더미) 변수와 연속형 변수를 구분해 해석 표현을 달리한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 원본 데이터프레임 (기본값: None). 주어지면 이분형/연속형을 구분해 표현한다.
        alpha (float): 유의성 판정에 사용할 유의수준 (기본값: 0.05).

    Returns:
        str: 독립변수별 해석 문장 불릿 리스트. `IPython.display.Markdown`으로 감싸 출력하면 좋다.
    """
    # --- 1) 해석 대상 결정 (상수항 제외) ---
    yname = fit.model.endog_names
    xnames = [name for name in fit.model.exog_names if name != "const"]

    # --- 2) 문장 템플릿 구성 (독립변수마다 반복 적용) ---
    line_template = (
        "- **{x}**의 회귀계수는 **B = {B}**, 오즈비는 **OR = {OR}**로 나타났으며, "
        "이는 **{y}**에 {sig} 요인임을 의미한다.      \n"
        "(**z = {z}**, **{p}**)      \n"
        "즉, {change} {y}가 1(사건 발생)이 될 오즈는 평균적으로 약 **{pct}% {direction}**하는 것으로 해석된다."
    )

    # --- 3) 독립변수별 해석 문장 생성 ---
    lines = []   # 독립변수별 문장(불릿)을 저장할 빈 리스트
    for x in xnames:
        B = fit.params[x]               # 비표준화 회귀계수(B)
        z = fit.tvalues[x]              # z-통계량
        p = fit.pvalues[x]              # 계수 유의확률
        OR = np.exp(B)                  # 오즈비 = exp(B)

        # 유의성 판정 (유의수준 기준)
        sig_word = "유의한" if p < alpha else "유의하지 않은"

        # p값 APA 표기 (앞자리 0 생략)
        if p < 0.001:   p_text = "p < .001"
        else:           p_text = f"p = {p:.3f}".replace("0.", ".")

        # 오즈비를 백분율 변화로 환산 (OR>1 증가, OR<1 감소)
        pct = abs((OR - 1) * 100)
        direction = "증가" if B > 0 else "감소"

        # 변화 표현: 로그 계열은 % 해석의 기준이 되는 값이 무엇인지가 핵심이다.
        # 변환하지 않은 변수만 이분형(더미)/연속형을 구분한다.
        is_binary = data is not None and data[x].nunique() <= 2
        if is_binary:   change = f"**{x}**에 해당하는 경우(기준 범주 대비)"
        else:           change = f"**{x}**가 1 증가할 때"

        # 하나의 독립변수 → 하나의 불릿 문장
        lines.append(line_template.format(
            x=x, B=round(B, 4), OR=round(OR, 4), y=yname, sig=sig_word,
            z=round(z, 2), p=p_text, change=change,
            pct=round(pct, 1), direction=direction,
        ))

    # --- 4) 해석 주의 각주 첨부 ---
    report = "\n".join(lines)
    report += (
        "\n\n> ※ 오즈비(OR)가 1보다 크면 사건 발생 오즈가 증가, 1보다 작으면 감소함을 뜻한다. "
        "유의확률이 유의수준보다 큰(=유의하지 않은) 변수는 효과가 통계적으로 확인되지 않았으므로 "
        "오즈비 해석에 주의한다. 더미변수의 오즈비는 '기준(drop_first로 제외된) 범주' 대비 값이다."
    )

    return report


def plot_odds(fit, data, palette=None, title=None, xlabel=None, ylabel=None,
              width=1280, height=None, save_path=None):
    """오즈비(Odds Ratio)를 가로 막대그래프로 시각화해 독립변수의 영향력을 보여준다.

    막대는 `report_variables`가 정렬해 둔 |B|(=|log OR|) 내림차순 그대로 위에서
    아래로 배치되며, 오즈비가 1보다 큰지(사건 발생 오즈 증가) 작은지(감소)에 따라
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
    rdf["부호"] = np.where(rdf["오즈비(OR)"] > 1, "+", "-")

    # 독립변수가 많을수록 막대가 촘촘해지므로, 변수 하나당 80px씩 세로 공간을 확보한다
    if height is None:
        height = len(rdf) * 80

    if height < 200:
        height = 200   # 최소 높이 200px

    # 부호별 기본 색상: 증가(+)은 파랑, 감소(-)은 빨강
    if palette is None:
        palette = {"+": "#0066ff", "-": "#ff3333"}

    # --- 2) 그래프 초기화 ---
    fig, ax = my_plot.init(width=width, height=height, title=title,
                           xlabel=xlabel if xlabel else "오즈비(Odds Ratio)",
                           ylabel=ylabel if ylabel else "독립변수")

    # --- 3) 가로 막대그래프 ---
    my_plot.barplot(rdf, x="오즈비(OR)", y="독립변수", hue="부호", palette=palette, ax=ax)

    # OR=1(영향 없음) 기준선
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.7)

    # --- 4) 막대 끝에 오즈비 값 표기 ---
    for i in rdf.index:
        orv = rdf.loc[i, "오즈비(OR)"]
        ax.text(x=orv, y=i, s=f"{orv:.2f}", va="center",
                ha="left" if orv >= 1 else "right", color="black")

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
    # --- 1) 모델 적합 (backward=True이면 유의하지 않은 변수가 없어질 때까지 반복) ---
    # 빈 줄 출력 (출력 결과의 여백을 위함)
    print()

    while True:
        fit = fit_model(data, y)

        if not backward:
            break   # 후진소거법이 아니면 반복문 종료

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
        display(Markdown("#### ▶︎ 모형 적합도"))   

        # 모형 적합도 해설                       
        display(Markdown(report_fitness(fit))) 
        
        display(Markdown("#### ▶︎ 독립변수 보고"))  
        # 오즈비 보고 표
        display(report_variables(fit, data))                      

        # 변수별 해석       
        display(Markdown(report_variables_text(fit, data=data, alpha=alpha)))  

    # --- 2) 시각화 ---
    if plot:
        display(Markdown("#### ▶︎ 오즈비 시각화"))
        plot_odds(fit, data, width=width)

    # --- 4) 최종 적합 모델 객체 반환 ---
    return fit


def plot_confusion(fit, threshold=0.5, palette=None, title=None,
                   size=640, save_path=None, ax=None):
    """혼동행렬(Confusion Matrix)을 정사각형 히트맵으로 시각화한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        threshold (float): 확률을 0/1로 분류하는 임계값 (기본값: 0.5).
        palette (str): 히트맵 색상 팔레트 이름 (기본값: None → "Blues").
        title (str): 그래프 제목 (기본값: None → "혼동행렬(Confusion Matrix)").
        size (int): 캔버스 한 변의 픽셀 크기. 가로·세로가 같다 (기본값: 640).
        save_path (str): 이미지 저장 경로 (기본값: None).
        ax: 그래프를 그릴 Axes 객체 (기본값: None → 캔버스를 새로 생성).
    """
    # --- 1) 실제값과 임계값 기준 예측 범주로 혼동행렬 구성 ---
    y_true = np.asarray(fit.model.endog).astype(int)
    y_pred = (np.asarray(fit.predict()) > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    cmdf = DataFrame(cm,
                     index=["실제 0 (Negative)", "실제 1 (Positive)"],
                     columns=["예측 0 (Negative)", "예측 1 (Positive)"])

    # --- 2) 시각화 ---
    # ax를 넘겨받으면 서브플롯의 한 칸에 그리고, 없으면 단독 캔버스를 만든다
    fig = None
    if ax is None:
        fig, ax = my_plot.init(width=size, height=size)

    ax.set_title(title if title else "혼동행렬(Confusion Matrix)",
                 fontsize=24, fontweight=500, pad=15)

    # square=True로 셀을 정사각형으로 맞추고, 값이 표기되므로 색상막대(cbar)는 생략한다
    # 빈도(정수)를 그대로 표기하도록 fmt="d" 사용
    my_plot.heatmap(data=cmdf, annot=True, fmt="d",
                    palette=palette if palette else "Blues",
                    annot_kws={"size": size*0.07},  # 글자크기를 그래프 크기의 7%로 설정
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
    proba = np.asarray(fit.predict())    # 모델의 예측값 --> 1이 될 확률  
    auc = roc_auc_score(y_true, proba)   # AUC 계산
    roc_fpr, roc_tpr, _ = roc_curve(y_true, proba) # ROC 좌표 (FPR,TPR,임계값)

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
    AUC는 구간별 판정("낮음"~"매우 우수")을, 진단오즈비는 (TP×TN)/(FP×FN)를 함께 출력한다.

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
    fallout = fp / (fp + tn) if (fp + tn) > 0 else 0.0      # 위양성률(FPR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 특이성(TNR = 1 - FPR)

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


def apply_recipe(data, recipe):
    """`fix_linear` 이 확정한 처방을 다른 데이터프레임에 그대로 재현한다.

    학습 때 사용한 중심화 평균·평행이동량을 그대로 써야 하므로,
    신규 데이터를 예측할 때는 반드시 이 함수로 파생변수를 만든다.

    Args:
        data: 처방을 적용할 데이터프레임. 처방 대상 원본 컬럼을 모두 포함해야 한다.
        recipe (dict): `fix_linear` 이 돌려준 `recipe_` 딕셔너리.
            {변수명: {'method': 'square'|'log', 'center': float, 'shift': float}}

    Returns:
        DataFrame: 파생변수가 추가되고 원본 컬럼이 제거된 데이터프레임.

    Raises:
        KeyError: 처방 대상 컬럼이 `data` 에 없는 경우.
        ValueError: 처방 방식이 'square'·'log' 가 아닌 경우.
    """
    tmp = data.copy()

    for col, rule in recipe.items():
        if col not in tmp.columns:
            raise KeyError(f"처방 대상 컬럼이 없습니다: '{col}'")

        method = rule["method"]

        if method == "square":
            # 중심화 후 제곱 — 평균은 반드시 학습 때 값을 재사용한다
            tmp[col + "_c"] = tmp[col] - rule["center"]
            tmp[col + "_c2"] = tmp[col + "_c"] ** 2
        elif method == "log":
            # 로그 변환 — 평행이동량도 학습 때 값을 재사용한다
            tmp[col + "_log"] = np.log(tmp[col] + rule["shift"])
        else:
            raise ValueError(f"지원하지 않는 처방 방식입니다: '{method}'")

        tmp = tmp.drop(columns=[col])

    return tmp


def test_linear(fit=None, data=None, yname=None, xnames=None, targets=None, alpha=0.05):
    """Box-Tidwell 검정으로 연속형 독립변수의 로짓 선형성을 점검한다.

    독립변수 x에 `x × ln(x)` 항을 하나씩 추가해 그 항이 유의한지 본다.
    유의하면(p < alpha) 로짓이 직선이 아니라는 뜻이므로 변환·제곱항·구간화가 필요하다.
    이분형(더미) 변수는 두 점을 잇는 선이 언제나 직선이므로 검정 대상에서 제외한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 회귀분석에 사용한 원본 데이터프레임. 독립변수와 종속변수를 모두 포함해야 한다.
        yname (str): 종속변수 컬럼명.
        xnames (list): 모델에 투입된 독립변수 이름 목록.
        targets (list): 검정할 변수 목록. None이면 `xnames` 중 연속형 전체 (기본값: None).
        alpha (float): 유의수준 (기본값: 0.05).
    """
    # --- 1) 검정 대상 결정 (상수항 제외, 연속형만) ---
    if yname is None:
        yname = fit.model.endog_names

    if xnames is None:
        xnames = []
        for name in fit.model.exog_names:
            if name != "const":
                xnames.append(name)

    if targets is None:
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
            "z": round(z, 4) if np.isfinite(z) else np.nan,
            "p-value": round(p, 4) if np.isfinite(p) else np.nan,
            "linearity": linearity,
            "위치이동": shifted,
            "result": conclusion,
            "독립변수": x,
        })

    # 결과표 생성 및 출력
    result_df = DataFrame(rows).set_index("독립변수")
    #display(result_df)
    return result_df


def fix_linear(data, y, alpha=0.05, allow_shifted_log=False, max_rounds=None, report=True):
    """로짓 선형성 위배를 한 변수씩 처방하고 재검정하는 루프를 자동으로 수행한다.

    Args:
        data: 독립변수와 0/1 종속변수를 모두 포함한 데이터프레임.
        y (str): 종속변수 컬럼명.
        alpha (float): 유의수준 (기본값: 0.05).
        allow_shifted_log (bool): 음수를 포함한 변수도 평행이동 후 로그 변환을 시도할지 여부.
            이동량이 자의적이어서 계수 해석이 무너지므로 기본값은 False (기본값: False).
        max_rounds (int): 최대 반복 횟수. None이면 연속형 독립변수 개수 (기본값: None).
        report (bool): 라운드별 진행 상황과 처리 이력표를 출력할지 여부 (기본값: True).

    Returns:
        처방이 모두 반영된 회귀분석 결과 객체. 아래 속성이 함께 붙는다.
            - `data_` (DataFrame): 파생변수가 반영된 데이터프레임
            - `recipe_` (dict): 변수별 처방 내역. `apply_recipe` 에 그대로 넘겨 쓴다
            - `history_` (DataFrame): 라운드별 처리 이력표
            - `unresolved_` (list): 제곱항·로그변환으로도 해소되지 않은 변수 목록

    Raises:
        KeyError: 종속변수 컬럼이 `data` 에 없는 경우.
        ValueError: alpha·max_rounds 가 유효하지 않은 경우.
    """
    # --- 1) 입력 검증 ---
    if y not in data.columns:
        raise KeyError(f"종속변수 컬럼이 없습니다: '{y}'")

    if not 0 < alpha < 1:
        raise ValueError(f"alpha 는 0과 1 사이여야 합니다: {alpha}")

    if max_rounds is not None and max_rounds < 1:
        raise ValueError(f"max_rounds 는 1 이상이어야 합니다: {max_rounds}")

    # --- 2) 초기 상태 준비 ---
    tmp = data.copy()
    fit = fit_model(tmp, y=y)               # 기준선 모형

    # 아직 처방하지 않은 연속형 독립변수 (이분형은 언제나 직선이므로 제외)
    pending = []
    for x in fit.model.exog_names:
        if x != "const" and tmp[x].nunique() > 2:
            pending.append(x)

    if max_rounds is None:          # max_rounds 가 지정되지 않으면 
        max_rounds = len(pending)   # 연속형 독립변수 개수만큼 반복한다

    recipe = {}         # 확정된 처방
    unresolved = []     # 처방으로 해소하지 못한 변수
    history = []        # 라운드별 이력

    # --- 3) 진단 → 처방 → 검증 → 재진단 루프 ---
    for rnd in range(1, max_rounds + 1):
        # --- 3-1) 진단: 아직 손대지 않은 변수만 다시 검정한다 ---
        # 이미 처방한 변수를 다시 넣으면 짝이 되는 제곱항 때문에 판정이 무의미하다
        targets = [x for x in pending if x not in unresolved]

        if not targets:
            break

        bt = test_linear(fit, data=tmp, yname=y, targets=targets, alpha=alpha)
        
        violated = bt[~bt["linearity"]]

        if violated.empty:
            if report:
                print(f"[라운드 {rnd}] 위배 변수 없음 → 루프 종료")
            break

        # --- 3-2) 위배가 가장 심한 변수 하나를 고른다 ---
        target = violated["z"].abs().idxmax()

        if report:
            print(f"[라운드 {rnd}] 위배 {len(violated)}종 → "
                  f"'{target}' 처방 (z = {violated.loc[target, 'z']})")

        # --- 3-3) 처방 ①: 중심화 후 제곱항 추가 ---
        center = float(tmp[target].mean())

        sq_data = tmp.copy()
        sq_data[target + "_c"] = sq_data[target] - center
        sq_data[target + "_c2"] = sq_data[target + "_c"] ** 2
        sq_data = sq_data.drop(columns=[target])

        # 적합이 실패하면 제곱항은 후보에서 탈락시킨다
        try:
            sq_fit = fit_model(sq_data, y=y)

            # 검증: 제곱항 자체가 유의한가 + 우도비 검정이 개선을 지지하는가
            sq_p = float(sq_fit.pvalues[target + "_c2"])
            lr_stat = 2 * (sq_fit.llf - fit.llf)
            lr_p = float(1 - chi2.cdf(lr_stat, 1))  # 늘어난 모수는 제곱항 1개
            sq_ok = bool(sq_p < alpha and lr_p < alpha)
        except Exception:
            sq_fit, sq_p, lr_p, sq_ok = None, np.nan, np.nan, False

        # --- 3-4) 처방 ②: 로그 변환 ---
        # 최솟값이 0 이하이면 ln 정의역을 맞추기 위해 평행이동해야 하는데,
        # 이동량이 자의적이어서 계수 해석이 무너지므로 기본값에서는 시도하지 않는다
        min_value = float(tmp[target].min())
        shift = 0.0 if min_value > 0 else -min_value + 1.0

        log_fit, log_ok = None, False

        if min_value > 0 or allow_shifted_log:
            log_data = tmp.copy()
            log_data[target + "_log"] = np.log(log_data[target] + shift)
            log_data = log_data.drop(columns=[target])

            try:
                log_fit = fit_model(log_data, y=y)

                # 검증: 로그 변환은 모수가 늘지 않으므로 우도비 검정을 쓸 수 없다.
                # 대신 AIC가 좋아졌고 그 변수의 선형성이 회복되었는지로 판정한다
                aic_better = bool(log_fit.aic < fit.aic)

                log_bt = test_linear(log_fit, data=log_data, yname=y,
                                     targets=[target + "_log"], alpha=alpha)
                log_linear = bool(log_bt.loc[target + "_log", "linearity"])
                log_ok = bool(aic_better and log_linear)
            except Exception:
                log_fit, log_ok = None, False

        # --- 3-5) 두 처방을 견주어 채택한다 ---
        # 제곱항은 단조 곡선도 근사하므로, '유의하지 않을 때만 로그'로 두면
        # 로그가 맞는 경우에도 제곱항이 먼저 채택된다.
        # 따라서 둘 다 검증을 통과하면 AIC가 낮은 쪽을 고른다
        if sq_ok and log_ok: method = "square" if sq_fit.aic <= log_fit.aic else "log"
        elif sq_ok:          method = "square"
        elif log_ok:         method = "log"
        else:                method = None

        if method == "square":
            tmp, fit = sq_data, sq_fit
            recipe[target] = {"method": "square", "center": center, "shift": None}
            pending.remove(target)

            history.append({
                "라운드": rnd, "변수": target, "처방": "제곱항",
                "제곱항 p": round(sq_p, 4), "검증 p": round(lr_p, 4),
                "AIC": round(sq_fit.aic, 2), "채택": True,
            })

            if report:
                print(f"    → 제곱항 채택 (제곱항 p = {sq_p:.4f}, "
                      f"우도비 p = {lr_p:.4e}, AIC = {sq_fit.aic:.2f})")

        elif method == "log":
            tmp, fit = log_data, log_fit
            recipe[target] = {"method": "log", "center": None, "shift": shift}
            pending.remove(target)

            history.append({
                "라운드": rnd, "변수": target, "처방": "로그변환",
                "제곱항 p": round(sq_p, 4) if np.isfinite(sq_p) else np.nan,
                "검증 p": None,
                "AIC": round(log_fit.aic, 2), "채택": True,
            })

            if report:
                reason = "제곱항 기각" if not sq_ok else "AIC 우위"
                print(f"    → 로그변환 채택 ({reason}, AIC = {log_fit.aic:.2f})")

        # --- 3-6) 둘 다 실패: 원본을 그대로 두고 미해소로 기록한다 ---
        else:
            unresolved.append(target)

            history.append({
                "라운드": rnd, "변수": target, "처방": "미해소",
                "제곱항 p": round(sq_p, 4) if np.isfinite(sq_p) else np.nan,
                "검증 p": None,
                "AIC": round(fit.aic, 2), "채택": False,
            })

            if report:
                print(f"    → 제곱항·로그변환 모두 기각 → 해결되지 않음")

    # --- 4) 결과 정리 ---
    history_df = DataFrame(history)

    if report:
        print()
        if not history_df.empty:
            display(history_df.set_index("라운드"))

        if unresolved:
            print(f"⚠ 미해소 변수: {unresolved}\n"
                  f"  제곱항·로그변환으로 담기지 않는 형태이므로 "
                  f"구간화나 스플라인이 필요하다는 신호입니다. 한계점으로 보고하세요.")

    fit.data_ = tmp
    fit.recipe_ = recipe
    fit.history_ = history_df
    fit.unresolved_ = unresolved

    return fit


def test_independent(fit):
    """Durbin-Watson으로 잔차의 독립성을 검정한다.

    로지스틱 회귀는 잔차가 0/1 구조라 일반 잔차를 쓸 수 없으므로 **피어슨 잔차**를 사용한다.
    본래 시계열 전용 검정이므로, 시간 순서가 없는 데이터에서는 통계량이 정상이어도
    실제로는 독립이 아닐 수 있다. 최종 판단은 **자료 수집 설계**로 해야 한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.

    Returns:
        DataFrame: Durbin-Watson 통계량과 독립성 판정을 담은 단일 행 결과표.
    """
    # --- 1) Durbin-Watson 통계량 계산 (피어슨 잔차 사용) ---
    dw = float(durbin_watson(fit.resid_pearson))

    # --- 2) DW값에 따른 독립성 판정 및 해석 ---
    if 1.5 <= dw <= 2.5:
        independence = True
        conclusion = "독립성 만족"
    elif dw < 1.5:
        independence = False
        conclusion = "독립성 위반 (양(+)의 자기상관)"
    else:
        independence = False
        conclusion = "독립성 위반 (음(-)의 자기상관)"

    # --- 3) 단일 행 결과표 구성 및 출력 ---
    result_df = DataFrame([{
            "statistic": round(dw, 4),
            "independence": independence,
            "result": conclusion,
        }],
        index=["Durbin-Watson"])

    return result_df



def fit_pipeline(data, y, columns=None, *,
                 # --- 1) 이상치 대체 (IQR 경계값, 행 삭제 없음) ---
                 outlier=False,             # 이상치 대체 수행 여부
                 # --- 2) 로짓 선형성 처방 재현 ---
                 recipe=None,               # fix_linear 이 확정한 처방 내역
                 # --- 3) 더미변수 인코딩 ---
                 encode=True,               # 명목형 독립변수의 더미 인코딩 수행 여부
                 # --- 4) 다중공선성 제거 (VIF) ---
                 vif=False,                 # 다중공선성 제거 수행 여부
                 vif_threshold=10.0,        # VIF 임계값
                 # --- 5) 정규화 ---
                 scale=False,               # 정규화 수행 여부
                 scale_method="standard",   # 사용할 스케일러 이름
                 # --- 6) 모델 적합 ---
                 backward=False,            # 후진소거법 수행 여부
                 alpha=0.05,                # 후진소거법의 변수 제거 기준 유의수준
                 # --- 기타 ---
                 name=None,                 # 모델을 구분할 이름. 결과 객체의 `name_` 속성이 된다
                 verbose=False):            # 단계별 전처리 내역 출력 여부
    """플래그로 지정한 전처리를 수행한 뒤 로지스틱 회귀모델을 적합한다.

    Args:
        data (DataFrame): 독립변수와 0/1 종속변수를 모두 포함하는 데이터프레임.
        y (str): 종속변수로 사용할 컬럼명.
        columns (list): 이상치 대체 대상이 되는 원본 연속형 독립변수 목록.
            None이면 숫자형 독립변수를 자동 선택한다 (기본값: None).
        outlier (bool): 이상치를 IQR 경계값으로 대체할지 여부 (기본값: False).
        recipe (dict): `fix_linear` 이 돌려준 `recipe_`. None이면 처방을 적용하지 않는다 (기본값: None).
        encode (bool): 명목형 독립변수의 더미 인코딩 수행 여부 (기본값: True).
        vif (bool): 다중공선성 제거 수행 여부 (기본값: False).
        vif_threshold (float): VIF 임계값 (기본값: 10.0).
        scale (bool): 정규화 수행 여부 (기본값: False).
        scale_method (str): 사용할 스케일러 이름 (기본값: 'standard').
        backward (bool): 후진소거법 수행 여부 (기본값: False).
        alpha (float): 후진소거법의 변수 제거 기준 유의수준 (기본값: 0.05).
        name (str): 모델을 구분할 이름. 결과 객체의 `name_` 속성이 된다 (기본값: None).
        verbose (bool): 단계별 전처리 내역 출력 여부 (기본값: False).

    Returns:
        적합이 완료된 회귀분석 결과 객체. 아래 속성이 함께 붙는다.
            - `data_` (DataFrame): 전처리가 끝난 데이터. `report_variables` 등이 사용한다
            - `recipe_` (dict): 적용한 로짓 선형성 처방. 신규 예측 시 `apply_recipe` 에 넘긴다
            - `name_` (str): 모델을 구분할 이름

    Raises:
        KeyError: 종속변수 컬럼이 `data` 에 없는 경우.
        ValueError: 결측치가 남아 있는 경우.
    """
    # --- 1) 종속변수 확인 및 작업본 준비 ---
    if y not in data.columns:
        raise KeyError(f"종속변수 '{y}'가 데이터프레임의 컬럼에 존재하지 않습니다.")

    df = data.copy()    # 원본을 보존하기 위해 복사본으로 작업

    # --- 2) 이상치 대체 대상 확정 ---
    # 지정이 없으면 숫자형 독립변수를 자동 선택한다
    if columns is None:
        columns = []
        for c in df.select_dtypes(include="number").columns:
            if c != y:
                columns.append(c)
    else:
        missing = []
        for c in columns:
            if c not in df.columns:
                missing.append(c)

        if missing:
            raise KeyError(f"데이터프레임에 존재하지 않는 컬럼입니다: {missing}")

    # --- 3) 이상치 대체 (IQR 경계값, 행 삭제 없음) ---
    # -> 파생변수를 만들기 전에 원본 변수에 적용해야 곡선이 잘리지 않는다
    if outlier and columns:
        df = my_prep.replace_outlier(df, columns=columns, verbose=verbose)

    # --- 4) 로짓 선형성 처방 재현 (중심화+제곱항 또는 로그변환) ---
    if recipe:
        df = apply_recipe(df, recipe)

    # --- 5) 더미변수 인코딩 (명목형 독립변수가 있을 때만 동작한다) ---
    if encode:
        nominal_cols = []
        for c in df.select_dtypes(include=["category", "object"]).columns:
            if c != y:
                nominal_cols.append(c)

        if nominal_cols:
            df = my_prep.dummies(df, columns=nominal_cols, verbose=verbose)

    # 현재 시점의 독립변수 목록 (파생변수·더미가 추가되었을 수 있다)
    alive = [c for c in df.columns if c != y]

    # --- 6) 다중공선성 제거 (VIF) ---
    if vif:
        df = my_prep.reduce_vif(df, columns=alive, threshold=vif_threshold,
                                verbose=verbose)
        alive = [c for c in df.columns if c != y]

    # --- 7) 정규화 ---
    if scale:
        df = my_prep.scaling(df, columns=alive, method=scale_method, verbose=verbose)

    # --- 8) 결측치 점검 ---
    na_cols = list(df.columns[df.isna().any()])

    if na_cols:
        raise ValueError(f"결측치가 있는 컬럼이 있습니다: {na_cols}\n"
                         f"데이터 품질 점검 단계에서 먼저 처리하세요.")

    # --- 9) 모델 적합 (backward=True 이면 후진소거법 수행) ---
    fit = auto_logit(df, y=y, backward=backward, alpha=alpha,
                     report=False, plot=False)

    # --- 10) 보고에 필요한 정보를 결과 객체에 붙여 반환 ---
    fit.data_ = df                      # report_variables 등이 β·VIF 계산에 사용한다
    fit.recipe_ = recipe or {}          # 신규 예측 시 apply_recipe 에 그대로 넘긴다
    fit.name_ = name                    # compare_models 가 채워 주기도 한다

    return fit


def compare_models(fits, metric="AUC", sub_metric="변수수", tolerance=0.05,
                   threshold=0.5, digits=4, report=True):
    """여러 로지스틱 모델의 분류 성능을 한 표로 정리해 좋은 순으로 정렬하고, 최고 모델을 반환한다.
    주 지표 1위와의 격차가 tolerance 이내면 '근소 격차 그룹'으로 묶어 보조 지표로 순서를 정한다.

    `my_ols.compare_models` 와 같은 방식으로 동작하되, 회귀 지표(RMSE·R²) 대신
    분류 지표(정확도·F1·AUC)와 모형 적합도 지표(Pseudo R²·AIC)로 비교한다.

    Args:
        fits (dict): {모델이름: 적합된 회귀분석 결과 객체} 형태의 딕셔너리.
        metric (str): 정렬 기준이 되는 주 성능평가지표 (기본값: 'AUC').
        sub_metric (str): 근소 격차 그룹 안에서 적용할 보조 지표. None이면 미사용 (기본값: '변수수').
        tolerance (float): 근소 격차로 판단할 주 지표의 상대격차. 0이면 순수 크기 비교 (기본값: 0.05).
        threshold (float): 확률을 0/1로 분류하는 임계값 (기본값: 0.5).
        digits (int): 표에 표시할 소수점 자릿수 (기본값: 4).
        report (bool): 성능 비교표를 화면에 출력할지 여부 (기본값: True).

    Returns:
        성능이 가장 좋은 모델의 회귀분석 결과 객체(표의 첫 행). 아래 속성이 함께 붙는다.
            - `name_` (str): 모델 이름. `fits` 의 키에서 채워진다
            - `score_table_` (DataFrame): 모델명을 인덱스로 하는 성능 비교표.
              성능이 좋은 모델이 위에 오며, 맨 끝에 1위 대비 상대격차인 `Gap(%)` 컬럼이 붙는다

    Raises:
        TypeError: `fits` 가 딕셔너리가 아니거나 값이 회귀분석 결과 객체가 아닌 경우.
        ValueError: `fits` 가 비었거나 지표 이름·tolerance·threshold 가 유효하지 않은 경우.
    """
    # --- 1) 지표별 '성능이 좋은 방향' 정의 (True = 값이 클수록 좋음) ---
    metrics = {
        "변수수": False,          # 같은 성능이라면 변수가 적은 모델이 간명하다
        "정확도": True,
        "정밀도": True,
        "재현율": True,
        "F1": True,
        "AUC": True,
        "Pseudo R²": True,
        "AIC": False,
    }

    # --- 2) 입력 검증 ---
    if not isinstance(fits, dict):
        raise TypeError(f"fits 는 딕셔너리여야 합니다: {type(fits).__name__}")

    if not fits:
        raise ValueError("비교할 모델이 없습니다.")

    for name, fit in fits.items():
        if not hasattr(fit, "prsquared"):
            raise TypeError(f"'{name}' 의 값이 로지스틱 회귀분석 결과 객체가 아닙니다: "
                            f"{type(fit).__name__}")

    for m in [metric, sub_metric]:
        if m is not None and m not in metrics:
            raise ValueError(f"지원하지 않는 지표입니다: '{m}' "
                             f"(사용 가능: {list(metrics.keys())})")

    if tolerance < 0:
        raise ValueError(f"tolerance 는 0 이상이어야 합니다: {tolerance}")

    if not 0 < threshold < 1:
        raise ValueError(f"threshold 는 0과 1 사이여야 합니다: {threshold}")

    # --- 3) 모델별 성능지표 계산 ---
    result = []

    for name, fit in fits.items():
        y_true = np.asarray(fit.model.endog).astype(int)    # 실제 종속변수(0/1)
        proba = np.asarray(fit.predict())                   # 1이 될 확률
        y_pred = (proba > threshold).astype(int)            # 임계값 기준 예측 범주

        result.append({
            "모델": name,
            "변수수": int(fit.df_model),        # 상수항을 제외한 독립변수 개수
            "정확도": accuracy_score(y_true, y_pred),
            "정밀도": precision_score(y_true, y_pred, zero_division=0),
            "재현율": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_true, proba),
            "Pseudo R²": fit.prsquared,
            "AIC": fit.aic,
        })

    rdf = DataFrame(result).set_index("모델")

    # --- 4) 1위 대비 주 지표의 상대격차 계산 ---
    # 지표마다 좋은 방향이 다르므로 metrics 에 기록해 둔 방향으로 최적값을 찾는다
    higher_is_better = metrics[metric]

    if higher_is_better:
        best = rdf[metric].max()
        diff = best - rdf[metric]      # 클수록 좋은 지표는 1위보다 작을수록 나쁘다
    else:
        best = rdf[metric].min()
        diff = rdf[metric] - best      # 작을수록 좋은 지표는 1위보다 클수록 나쁘다

    # AIC 처럼 값이 음수인 지표도 있으므로 최적값의 절댓값을 분모로 삼는다.
    # 최적값이 0 이면 나눌 수 없으므로 격차를 값의 차이 그대로 본다
    if best != 0:
        denominator = abs(best)
    else:
        denominator = 1.0

    rdf["Gap(%)"] = (diff / denominator * 100).round(2)    # 양수일수록 1위보다 나쁨

    # --- 5) 근소 격차 그룹을 먼저 정렬하고 나머지를 뒤에 붙인다 ---
    # 주 지표가 사실상 비슷한(격차가 tolerance 이내인) 모델끼리는 보조 지표로 순서를 정한다
    close = rdf["Gap(%)"] <= tolerance * 100

    by = [metric]
    ascending = [not higher_is_better]

    if sub_metric:
        # 보조 지표를 앞에 두어야 근소 격차 그룹 안에서 우선 적용된다
        by.insert(0, sub_metric)
        ascending.insert(0, not metrics[sub_metric])

    front = rdf[close].sort_values(by=by, ascending=ascending)
    back = rdf[~close].sort_values(by=[metric], ascending=[not higher_is_better])

    score_table = concat([front, back]).round(digits)

    # --- 6) 성능표 출력 ---
    if report:
        display(score_table)

    # 각 모델에 딕셔너리 키를 이름으로 새겨 둔다 (직접 지정한 name_ 이 없을 때만)
    for model_name, fit in fits.items():
        if getattr(fit, "name_", None) is None:
            fit.name_ = model_name

    # 표는 성능순으로 정렬되어 있으므로 첫 행이 곧 최고 모델이다
    best = fits[score_table.index[0]]
    best.score_table_ = score_table

    return best


def report_threshold(fit, thresholds=None, digits=3):
    """임계값을 바꿔가며 분류 성능이 어떻게 달라지는지 한 표로 정리한다.

    임계값 0.5는 관례일 뿐이므로, 정밀도와 재현율 중 무엇이 중요한지에 따라
    조정한 결과를 비교해 목적에 맞는 값을 고르는 데 사용한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        thresholds (list): 비교할 임계값 목록 (기본값: [0.3, 0.4, 0.5, 0.6, 0.7]).
        digits (int): 표에 표시할 소수점 자릿수 (기본값: 3).
        report (bool): 비교표를 화면에 출력할지 여부 (기본값: True).

    Returns:
        DataFrame: 임계값을 인덱스로 하는 성능 비교표.
            정확도·정밀도·재현율·F1 과 놓친 1(FN)·잘못 잡은 0(FP) 건수를 담는다.

    Raises:
        ValueError: 임계값 목록이 비었거나 0~1 범위를 벗어난 값이 있는 경우.
    """
    # --- 1) 입력 검증 ---
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    if not len(thresholds):
        raise ValueError("비교할 임계값이 없습니다.")

    for t in thresholds:
        if not 0 < t < 1:
            raise ValueError(f"임계값은 0과 1 사이여야 합니다: {t}")

    # --- 2) 실제값·예측확률 준비 ---
    y_true = np.asarray(fit.model.endog).astype(int)    # 실제 종속변수(0/1)
    proba = np.asarray(fit.predict())                   # 1이 될 확률

    # --- 3) 임계값별 성능 계산 ---
    rows = []

    for t in thresholds:
        y_pred = (proba > t).astype(int)                # 임계값 t 로 이진화
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        rows.append({
            "임계값": t,
            "정확도": accuracy_score(y_true, y_pred),
            "정밀도": precision_score(y_true, y_pred, zero_division=0),
            "재현율": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "놓친 1(FN)": int(fn),      # 임계값을 올릴수록 늘어난다
            "잘못 잡은 0(FP)": int(fp),  # 임계값을 내릴수록 늘어난다
        })

    result = DataFrame(rows).set_index("임계값").round(digits)

    return result