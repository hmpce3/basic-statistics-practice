import os
import numpy as np
import seaborn as sb
from IPython.display import display, Markdown
from pandas import DataFrame, concat
from statsmodels.api import add_constant, OLS
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.api import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import zscore, probplot, shapiro, kstest
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from . import my_plot
from . import my_stats
from . import my_prep
from . import my_qtcheck


def fit_model(data, y, summary = False):
    """종속변수를 제외한 모든 컬럼을 독립변수로 삼아 OLS 회귀모델을 적합한다(절편 자동 추가).

    Args:
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        y: 종속변수로 사용할 컬럼명.
        summary (bool): 적합 모델의 요약 통계량 출력 여부 (기본값: False).

    Returns:
        적합이 완료된 회귀분석 결과 객체.
    """
    if y not in data.columns:
        raise KeyError(f"종속변수 '{y}'가 데이터프레임의 컬럼에 존재하지 않습니다.")

    x = data.drop(columns=[y])      # 독립변수 데이터프레임 생성
    y_series = data[y]              # 종속변수 시리즈 생성
    x_input = add_constant(x)       # 독립변수에 절편(상수항) 추가
    model = OLS(y_series, x_input)  # OLS 모델 객체 생성
    fit = model.fit()               # 모델 적합(Fit)

    if summary:
        print(fit.summary())        # 적합된 모델의 요약 통계량 출력 여부 확인

    return fit                      # 적합된 모델 객체(분석 결과) 반환


def predict(fit, new_data):
    """적합된 회귀모델을 이용해 새로운 데이터에 대한 예측값을 계산한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        new_data: 예측에 사용할 새로운 데이터프레임. 독립변수 컬럼만 포함해야 한다.

    Returns:
        DataFrame: 예측값을 담은 데이터프레임 (컬럼명 'pred').
    """
    # 새로운 데이터에 절편(상수항) 추가
    new_data_with_const = add_constant(new_data)

    # 예측값 계산
    predictions = fit.predict(new_data_with_const)

    # 예측값을 DataFrame으로 반환
    return DataFrame(predictions, columns=["pred"])


def test_linear(fit, alpha=0.05, plot=True, palette=None, title=None,
                xlabel=None, ylabel=None, width=1280, height=640, save_path=None):
    """Ramsey RESET(power=2)으로 잔차의 선형성(모형 설정 오류)을 검정한다.

    고차항이 유의하면(p < alpha) 직선으로 잡지 못한 곡선 관계가 남아 있다는 뜻이다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        alpha (float): 유의수준 (기본값: 0.05).
        plot (bool): 적합값-잔차 산점도(lowess 추세선 포함)를 시각화할지 여부 (기본값: True).
        palette (str): 산점도 점 색상에 사용할 팔레트 이름. None이면 기본색 (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 라벨 (기본값: None → "적합값(예측값)").
        ylabel (str): y축 라벨 (기본값: None → "잔차(residual)").
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
        save_path (str): 그래프 저장 경로 (기본값: None).
    """
    # --- 1) Ramsey RESET 검정 (고차항 power=2, F-검정) ---
    reset_res = linear_reset(fit, power=2, use_f=True)  # F-검정 수행
    fvalue = float(reset_res.fvalue)    # F 통계량
    pvalue = float(reset_res.pvalue)    # p-value
    linearity = bool(pvalue >= alpha)   # 선형성 가정 충족 여부 (True, False)

    # --- 2) 결과 해석 문자열 ---
    if linearity:
        conclusion = "귀무가설 채택 → 선형성 위배 근거 없음"
    else:
        conclusion = "대립가설 채택 → 선형성 위배(곡선 관계 존재)"

    # --- 3) 단일 행 결과표 구성 ---
    result_df = DataFrame( [{
            "statistic": round(fvalue, 4),
            "p-value": round(pvalue, 4),
            "linearity": linearity,
            "result": conclusion,
        }], index=["Ramsey RESET"])

    display(result_df)  # 결과표 출력

    # --- 4) 시각화: 적합값 대비 잔차 산점도 + lowess 추세선 ---
    if plot:
        # 팔레트가 지정되면 첫 번째 색을 산점도 점 색상으로 사용
        point_color = sb.color_palette(palette)[0] if palette else "#328CC1"

        plot_df = DataFrame({"y_pred": fit.fittedvalues, "resid": fit.resid})

        fig, ax = my_plot.init(width=width, height=height, title=title,
                               xlabel=xlabel if xlabel else "적합값(예측값)",
                               ylabel=ylabel if ylabel else "잔차(residual)")
                               
        # 잔차=0 기준선(파란 점선)
        my_plot.lineplot(x=[plot_df["y_pred"].min(), plot_df["y_pred"].max()], 
                         y=[0, 0], color="blue", linestyle="--", ax=ax)

        # 잔차 산점도 + lowess(비선형) 추세선(빨강)
        sb.regplot(data=plot_df, x="y_pred", y="resid", lowess=True,
            scatter_kws={"color": point_color, "edgecolor": "#ffffff", "alpha": 0.8},
            line_kws={"color": "red"}, ax=ax)

        my_plot.show(save_path=save_path)


def test_normal(fit, alpha=0.05, plot=True, palette=None, width=1280, height=640):
    """잔차의 정규성을 두 가지 방법으로 검정하고 진단 결과를 순서대로 출력한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        alpha (float): 유의수준 (기본값: 0.05).
        plot (bool): Q-Q 플롯과 √MSE 잔차도를 함께 그릴지 여부 (기본값: True).
        palette (str): 그래프 색상에 사용할 팔레트 이름. None이면 기본색 (기본값: None).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
    """
    # --- 1) 잔차 추출 및 표본수에 따른 검정 선택 ---
    resid = fit.resid                   # 잔차(residual) 추출
    n = len(resid)                      # 표본수(n) 확인

    if n < 30:
        method = "Shapiro-Wilk"         # 표본수가 30 미만이면 Shapiro-Wilk 검정 사용
        s, p = shapiro(resid)           # Shapiro-Wilk 검정 통계량 및 p값
    else:
        method = "Kolmogorov-Smirnov"   # 표본수가 30 이상이면 Kolmogorov-Smirnov 검정 사용
        # 표본 평균·표준편차로 표준화한 뒤 표준정규분포(N(0,1))와 비교
        # (kstest에 loc/scale을 넘기는 방식은 scipy 버전에 따라 오류가 발생하므로
        #  표준화 방식으로 동일한 검정을 수행)
        mu = resid.mean()               # 잔차 평균
        sigma = resid.std(ddof=1)       # 잔차 표준편차(표본분산)
        z = (resid - mu) / sigma        # 잔차 표준화
        s, p = kstest(z, "norm")        # 표준정규분포와 비교한 K-S 검정 통계량 및 p값

    s = float(s)                        # 검정 통계량
    p = float(p)                        # p-value
    normality = bool(p >= alpha)        # 정규성 가정 충족 여부 (True, False)

    # --- 2) 검정 통계량 결과표 ---
    test_df = DataFrame(
        [{
            "statistic": round(s, 4),
            "p-value": round(p, 4),
            "normality": normality,
            "result": ("귀무가설 채택 → 정규성 만족" if normality
                       else "대립가설 채택 → 정규성 위배"),
        }],
        index=[method],
    )

    display(test_df)

    # --- 3) Q-Q 플롯 ---
    if plot:
        # 팔레트가 지정되면 첫 번째 색을 Q-Q 기준선 색상으로 사용
        line_color = sb.color_palette(palette)[0] if palette else "red"

        # 잔차를 z-score 표준화한 뒤 Q-Q 플롯용 분위수 계산
        (theoretical, sample), _ = probplot(zscore(resid))

        # Q-Q 플롯용 데이터프레임 생성
        qq_df = DataFrame({"qq_x": theoretical, "qq_y": sample})

        # Q-Q 플롯 그리기
        my_plot.lmplot(
            data=qq_df, x="qq_x", y="qq_y",
            linecolor=line_color, linestyle="--",
            xlabel="이론 분위수(Theoretical Quantiles)",
            ylabel="표본 분위수(Sample Quantiles)",
            width=width, height=height,
        )

    # --- 4) √MSE 구간 규칙(68-95-99.7) 판정 ---
    sqrt_mse = float(np.sqrt(fit.mse_resid))  # 잔차 표준편차 추정치(√MSE)
    expected = [0.68, 0.95, 0.997]            # ±1·±2·±3√MSE 구간의 정규분포 기대 비율
    ratios = []                               # 구간별 실제 포함 비율(%) — 잔차도 주석용
    mse_rows = []                             # 구간별 판정 상세 (판정표용)
    mse_pass = []                             # 구간별 규칙 충족 여부

    for k, exp in zip((1, 2, 3), expected):
        # 해당 구간에 포함된 잔차의 실제 비율
        actual = float(((resid > -k * sqrt_mse) & (resid < k * sqrt_mse)).sum() / n)
        ratios.append(actual * 100)           # 구간별 실제 포함 비율(%)를 리스트에 저장
        # 기대 비율의 표준오차(±2SE)로 허용 범위 산출 후 [0, 1]로 클리핑
        se = np.sqrt(exp * (1 - exp) / n)     # 표준오차(SE) 계산
        lo = max(0.0, exp - 2 * se)           # 허용 범위 하한
        hi = min(1.0, exp + 2 * se)           # 허용 범위 상한
        ok = bool(lo <= actual <= hi)         # 실제 비율이 허용 범위 안에 있는지 여부
        mse_pass.append(ok)                   # 구간별 규칙 충족 여부를 리스트에 저장
        mse_rows.append({                     # 구간별 판정 상세 내용를 딕셔너리로 구성
            "구간": f"±{k}√MSE",
            "기대(%)": round(exp * 100, 1),
            "허용범위(%)": f"{lo * 100:.0f}~{hi * 100:.0f}",
            "실제(%)": round(actual * 100, 2),
            "판정": "충족" if ok else "위배",
        })

    mse_df = DataFrame(mse_rows).set_index("구간")  # 구간별 판정 상세 결과표 생성
    display(mse_df)                                # 구간별 판정 상세 결과표 출력

    mse_rule = bool(all(mse_pass))            # 세 구간 모두 충족해야 규칙상 정규성 부합
    print(f"√MSE = {sqrt_mse:.2f} · 구간 규칙 판정: {'정규성 부합' if mse_rule else '정규성 위배'}")

    # --- 5) √MSE 잔차도 (적합값 대비 잔차 + ±√MSE 구간) ---
    if plot:
        # 팔레트가 지정되면 3색을 뽑아 ±√MSE 구간 색상으로, 가운데 색을 산점도 색상으로 사용
        band_colors = (sb.color_palette(palette, n_colors=3) if palette
                       else ["#0B3C5D", "#328CC1", "#D9EAF7"])
        point_color = band_colors[1] if palette else "#328CC1"

        # √MSE 잔차도를 위한 데이터프레임 생성
        plot_df = DataFrame({"y_pred": fit.fittedvalues, "resid": fit.resid})
        
        # 적합값 대비 잔차 산점도
        fig, ax = my_plot.init(width=width, height=height,
                               xlabel="적합값(예측값)", ylabel="잔차(residual)")
        sb.scatterplot(data=plot_df, x="y_pred", y="resid",
                       color=point_color, edgecolor="#ffffff", ax=ax)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.6)

        # ±1·±2·±3√MSE 구간 표시 및 포함 비율 주석
        for i, c in enumerate(band_colors):
            k = i + 1
            y_pos = k * sqrt_mse
            ax.axhline(y=y_pos, color=c, linestyle="--", alpha=0.6)
            ax.axhline(y=-y_pos, color=c, linestyle="--", alpha=0.6)
            ax.text(x=1.02, y=0.5 + 0.12 * k, s=f"+{k} √MSE = {ratios[i]:.2f}%",
                    transform=ax.transAxes, ha="left", va="center", fontsize=11, color=c)
            ax.text(x=1.02, y=0.5 - 0.12 * k, s=f"-{k} √MSE = {ratios[i]:.2f}%",
                    transform=ax.transAxes, ha="left", va="center", fontsize=11, color=c)
        
        my_plot.show()


def test_equalvar(fit, alpha=0.05):
    """잔차의 등분산성을 검정하고 결과표를 출력한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        alpha (float): 유의수준 (기본값: 0.05).
    """
    # --- 1) Breusch-Pagan 검정 (LM/F 통계량) ---
    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(fit.resid, fit.model.exog)
    f_p = float(f_p)
    homoscedasticity = bool(f_p >= alpha)            # alpha 기준 등분산 채택 여부

    # --- 2) 두 임계값(alpha, strict_alpha)을 비교한 결과 해석 문자열 ---
    if f_p <= alpha:
        conclusion = f"대립가설 채택 → 등분산 아님"
    else:
        conclusion = f"귀무가설 채택 → 등분산성 만족"

    # --- 3) 단일 행 결과표 구성 및 반환 ---
    result_df = DataFrame([{
            "LM statistic": round(float(lm_stat), 4),
            "LM p-value": round(float(lm_p), 4),
            "F statistic": round(float(f_stat), 4),
            "F p-value": round(f_p, 4),
            "homoscedasticity": homoscedasticity,
            "result": conclusion,
        }], index=["Breusch-Pagan"])

    display(result_df)  # 결과표 출력


def test_independent(fit):
    """Durbin-Watson으로 잔차의 독립성을 검정한다.

    본래 시계열 전용 검정이므로, 시간 순서가 없는 데이터에서는 위배되어도 무시해도 되는 경우가 많다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
    """
    # --- 1) Durbin-Watson 통계량 계산 ---
    dw = float(durbin_watson(fit.resid))

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
    result_df = DataFrame( [{
            "statistic": round(dw, 4),
            "independence": independence,
            "result": conclusion,
        }],
        index=["Durbin-Watson"])

    display(result_df)  # 결과표 출력


def report_fitness(fit, log_y=False, log_x=None, log1p_y=False, log1p_x=None,
                   reflect_y=False, reflect_x=None):
    """적합된 회귀모델의 모형 적합도를 학술 보고 형식의 문장으로 생성해 반환한다.

    변환한 변수는 log(...)/log1p(...)/log1p(max-...)로 표기해 실제 적합한 모형을 그대로 드러낸다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        log_y (bool): 종속변수에 로그변환(log)을 적용했는지 여부 (기본값: False).
        log_x (list | None): log 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        log1p_y (bool): 종속변수에 log1p 변환을 적용했는지 여부 (기본값: False).
        log1p_x (list): log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        reflect_y (bool): 종속변수에 반사 후 log1p 변환을 적용했는지 여부 (기본값: False).
        reflect_x (list): 반사 후 log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).

    Returns:
        str: 모형 적합도 보고 문장.
    """
    # --- 1) 변수 라벨 구성 ---
    # log_x, log1p_x, reflect_x는 정확한 독립변수 이름 리스트로 전달된다고 가정한다.
    log_x = log_x or []
    log1p_x = log1p_x or []
    reflect_x = reflect_x or []

    # 상수항(const)을 제외한 독립변수 이름 (위치가 아니라 이름으로 걸러낸다)
    xnames = []
    for name in fit.model.exog_names:
        if name != "const":
            xnames.append(name)

    # 변환이 적용된 변수는 문장에 log(...)/log1p(...)로 표기한다.
    # 반사 변환은 log1p 와 식이 다르므로(대소가 뒤집힌다) 라벨도 구분해 적는다.
    yname = fit.model.endog_names
    if reflect_y:   ylabel = f"log1p(max-{yname})"
    elif log1p_y:   ylabel = f"log1p({yname})"
    elif log_y:     ylabel = f"log({yname})"
    else:           ylabel = yname

    xlabels = []   # 독립변수별 표기 라벨
    for x in xnames:
        if x in reflect_x:  xlabels.append(f"log1p(max-{x})")
        elif x in log1p_x:  xlabels.append(f"log1p({x})")
        elif x in log_x:    xlabels.append(f"log({x})")
        else:               xlabels.append(x)

    xlabel = ", ".join(xlabels)

    # --- 2) 유의확률 구간 표기 변환 ---
    if fit.f_pvalue < 0.001:
        alpha = "< 0.001"
    elif fit.f_pvalue < 0.01:
        alpha = "< 0.01"
    elif fit.f_pvalue < 0.05:
        alpha = "< 0.05"
    else:
        alpha = "≥ 0.05"

    # --- 3) 문장 템플릿 구성 ---
    # (summary() 표를 파싱하지 않고 fit 속성에서 값을 직접 가져오며, 표시값과 동일하게
    #  보이도록 round()로 자리수만 맞춘다. Durbin-Watson은 가중잔차(wresid) 기반 계산값.)
    template = (
        "**Note. n = {n}. "
        "F({df_model}, {df_resid}) = {f_value}, "
        "p {alpha}, "
        "R² = {r_squared}, "
        "Adj.R² = {adj_r_squared}, "
        "Durbin-Watson = {durbin_watson}**\n\n"
        "{Y}를 종속변수로, {X}(을)를 독립변수로한 {type}회귀분석 결과, "
        "모형은 통계적으로 {result}.\n\n"
        "> F({df_model}, {df_resid}) = {f_value}, p {alpha}, R² = {r_squared}.\n\n"
        "즉, {X}는 {Y}의 약 {r_squared_percent}%를 설명하는 것으로 나타났다."
    )

    # --- 4) 회귀유형, 유의수준 판별 ---
    # 독립변수 개수로 회귀분석 유형 판별
    if len(xnames) == 1:    reg_type = "단순선형"
    else:                   reg_type = "다중선형"

    # 유의수준(0.05) 기준 모형의 통계적 유의성 판정
    if fit.f_pvalue < 0.05: result = "유의하였다"
    else:                   result = "유의하지 않았다"

    # --- 5) 문장 템플릿 값 치환 ---
    report = template.format(
        n=int(fit.nobs),
        df_model=int(fit.df_model),
        df_resid=int(fit.df_resid),
        f_value=round(fit.fvalue, 2),
        alpha=alpha,
        r_squared=round(fit.rsquared, 3),
        adj_r_squared=round(fit.rsquared_adj, 3),
        durbin_watson=round(durbin_watson(fit.wresid), 3),
        Y=ylabel,
        X=xlabel,
        type=reg_type,
        result=result,
        r_squared_percent=round(fit.rsquared * 100, 2),
    )

    # --- 6) 결과 리턴 ---
    return report


def report_variables(fit, data, hc3=False):
    """적합된 회귀모델의 독립변수별 회귀계수 보고표를 데이터프레임으로 생성해 반환한다.

    β·공차·VIF 계산에 원본 데이터의 표준편차가 필요하므로 `data`를 함께 받는다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 회귀분석에 사용한 원본 데이터프레임. 독립변수와 종속변수를 모두 포함해야 한다.
        hc3 (bool): True이면 HC3 로버스트 표준오차를 사용한다 (기본값: False).

    Returns:
        DataFrame: 종속변수·독립변수·B·표준오차·β·t·유의확률·공차·VIF 컬럼의 보고표.
            hc3가 True이면 일반 OLS와 로버스트(HC3)의 표준오차·t·유의확률이 나란히 배치된다.
    """
    # --- 1) 대상 변수 확인 및 VIF 계산 ---
    yname = fit.model.endog_names       # 종속변수 이름
    # 상수항(const)을 포함한 전체 변수 이름 순서 (위치 인덱스 계산에 사용)
    exog_names = list(fit.model.exog_names)
    # 상수항(const)을 제외한 독립변수 이름
    xnames = [name for name in exog_names if name != "const"]

    # 독립변수 전체를 대상으로 VIF를 한 번에 계산 (상수항 제외한 결과가 반환된다)
    vif = my_stats.compute_vif(data, columns=xnames)

    # 일반 OLS 통계량을 위치 인덱스로 접근하기 위해 배열로 변환
    params = np.asarray(fit.params)
    bse = np.asarray(fit.bse)
    tvalues = np.asarray(fit.tvalues)
    pvalues = np.asarray(fit.pvalues)

    # --- (신규) cov_type 지정 시 로버스트 표준오차·t·유의확률을 별도로 계산 ---
    # 일반값을 덮어쓰지 않고 비교용으로 따로 보관한다. t는 정의상 계수/표준오차이므로
    # 표준오차가 로버스트로 바뀌면 t도 함께 바뀐다(t = B / 로버스트 SE). 유의확률도 이 로버스트
    # t에서 나온다. 따라서 로버스트 SE·t·유의확률을 한 세트로 가져온다. 로버스트 결과 객체 역시
    # 이름 없는 배열로 반환되므로 동일하게 위치 인덱스로 접근한다.
    if hc3:
        robust = fit.get_robustcov_results(cov_type="HC3")
        rob_bse = np.asarray(robust.bse)
        rob_tvalues = np.asarray(robust.tvalues)
        rob_pvalues = np.asarray(robust.pvalues)

    # --- 2) 독립변수별 계수·통계량 정리 ---
    variables = []   # 독립변수별 보고 내용을 저장할 빈 리스트
    for x in xnames:
        # 미리 계산해 둔 VIF 표에서 해당 독립변수의 값을 조회
        vif_value = vif.loc[x, "VIF"]
        i = exog_names.index(x)         # 상수항을 포함한 전체 순서에서의 위치
        b = float(params[i])            # 비표준화 회귀계수(B)
        # 표준화 회귀계수(β) = B × (독립변수 표준편차 / 종속변수 표준편차)
        beta = b * (data[x].std(ddof=1) / data[yname].std(ddof=1))

        if hc3:
            # 로버스트 비교 형식: B(양쪽 공유) + 일반(SE·t·유의확률) + 로버스트(SE·t·유의확률)를
            # 대칭으로 배치한다. 각 방식의 SE·t·유의확률이 한 세트로 서로 대응된다.
            row = {
                "종속변수": yname,                  # 종속변수 이름
                "독립변수": x,                      # 독립변수 이름
                "B": b,                            # 비표준화 회귀계수(양쪽 동일)
                "표준오차": bse[i],                 # 일반 OLS 표준오차
                "표준오차(HC3)": rob_bse[i],        # 로버스트 표준오차
                "β": beta,                         # 표준화 회귀계수
                "t": tvalues[i],                   # 일반 OLS t
                "t(HC3)": rob_tvalues[i],          # 로버스트 t (= B / 로버스트 SE)
                "유의확률": pvalues[i],             # 일반 OLS 유의확률
                "유의확률(HC3)": rob_pvalues[i],    # 로버스트 유의확률
                "공차": 1 / vif_value,             # 공차(Tolerance = 1/VIF)
                "VIF": vif_value,                  # 분산팽창계수
            }
        else:
            # 기본(일반 OLS) 보고 형식
            row = {
                "종속변수": yname,            # 종속변수 이름
                "독립변수": x,                # 독립변수 이름
                "B": b,                      # 비표준화 회귀계수(B)
                "표준오차": bse[i],           # 계수 표준오차
                "β": beta,                   # 표준화 회귀계수(β)
                "t": tvalues[i],             # t-통계량
                "유의확률": pvalues[i],       # 계수 유의확률
                "공차": 1 / vif_value,        # 공차(Tolerance = 1/VIF)
                "VIF": vif_value,             # 분산팽창계수
            }

        variables.append(row)

    # --- 3) 독립변수별 보고표 생성 및 반환 ---
    vdf = DataFrame(variables)

    # β의 절대값으로 내림차순 정렬후 리턴 (영향력이 큰 변수가 위로 오도록)
    vdf = vdf.sort_values("β", key=abs, ascending=False).reset_index(drop=True)
    return vdf


def report_variables_text(fit, log_y=False, log_x=None, log1p_y=False, log1p_x=None,
                          reflect_y=False, reflect_x=None, hc3=False):
    """독립변수별 회귀계수 해석 문장을 markdown 불릿 리스트로 생성해 반환한다.

    반사 변환(log(1+max-x))은 % 해석의 기준이 반사값 (1+max-변수)이고, 반사한 변수가
    홀수 개면 원 변수 기준 방향이 반대가 된다. 효과 크기 계산식은 log1p와 같다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        log_y (bool): 종속변수에 log 변환을 적용했는지 여부 (기본값: False).
        log_x (list): log 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        log1p_y (bool): 종속변수에 log1p 변환을 적용했는지 여부 (기본값: False).
        log1p_x (list): log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        reflect_y (bool): 종속변수에 반사 후 log1p 변환을 적용했는지 여부 (기본값: False).
        reflect_x (list): 반사 후 log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        hc3 (bool): True이면 HC3 로버스트 표준오차를 사용한다 (기본값: False).

    Returns:
        str: 독립변수별 해석 문장 불릿 리스트.
    """
    # --- 1) 종속변수 정보 ---
    yname = fit.model.endog_names       # 종속변수 이름

    # 종속변수의 변환 종류 (구체적인 변환부터 확인한다)
    if reflect_y:   y_kind = "reflect"
    elif log1p_y:   y_kind = "log1p"
    elif log_y:     y_kind = "log"
    else:           y_kind = "none"

    y_pct = y_kind != "none"   # 로그 계열이면 비율(%) 해석 대상이다

    # % 해석의 대상: 반사는 (1+max-y), log1p는 (1+y), 그 외는 y
    if y_kind == "reflect":  y_target = f"**(1+max-{yname})**"
    elif y_kind == "log1p":  y_target = f"**(1+{yname})**"
    else:                    y_target = yname

    # --- 2) 독립변수 정보 ---
    # log_x, log1p_x, reflect_x는 정확한 독립변수 이름 리스트로 전달된다고 가정한다.
    log_x = log_x or []
    log1p_x = log1p_x or []
    reflect_x = reflect_x or []

    # 상수항(const)을 제외한 독립변수 이름 (위치가 아니라 이름으로 걸러낸다)
    xnames = []
    for name in fit.model.exog_names:
        if name != "const":
            xnames.append(name)

    # 독립변수의 변환 종류를 하나의 값으로 판별한다 (종속변수와 같은 순서로 확인한다)
    def kind_of(name):
        if name in reflect_x:   return "reflect"
        if name in log1p_x:     return "log1p"
        if name in log_x:       return "log"
        return "none"

    # --- 3) 그 밖의 정보 (자유도·로버스트 통계량) ---
    df_resid = int(fit.df_resid)        # t분포 자유도(잔차 자유도)

    # hc3=True이면 로버스트 표준오차 기반 t·유의확률로 교체한다.
    # 회귀계수(B)는 그대로이고 표준오차만 이분산에 강건한 HC3로 바뀌는데,
    # t = B / 로버스트 SE 이므로 t와 유의확률도 한 세트로 함께 바뀐다.
    # 로버스트 결과 객체는 이름 없는 배열을 반환하므로 위치 인덱스로 접근한다.
    if hc3:
        robust = fit.get_robustcov_results(cov_type="HC3")
        rob_tvalues = np.asarray(robust.tvalues)
        rob_pvalues = np.asarray(robust.pvalues)

    # --- 4) 문장 템플릿 구성 (독립변수마다 반복 적용) ---
    line_template = (
        "- **{x}**의 회귀계수는 **B = {B}**으로 나타났으며, "
        "이는 **{y}**에 {sig} 요인임을 의미한다. "
        "(**t({df}) = {t}**, **{p}**)      \n"
        "즉, {effect} 것으로 해석된다.{note}"
    )
    effect_template = "{x_change} {y_target}는 평균적으로 {approx}**{mag}{unit} {direction}**하는"
    # 반사 변환이 끼면 위 문장은 반사값 기준이므로, 원 변수 기준의 방향을 짧게 덧붙인다
    note_template = " (원 변수 기준: **{x}가 클수록 {y} {orig_direction}**)"
    opposite = {"증가": "감소", "감소": "증가"}   # 반사로 뒤집힌 방향을 되돌릴 때 쓴다

    # --- 5) 독립변수별 해석 문장 생성 ---
    lines = []   # 독립변수별 문장(불릿)을 저장할 빈 리스트
    for x in xnames:
        # 5-1) 계수와 검정 통계량 추출
        x_kind = kind_of(x)             # none / log / log1p / reflect
        x_pct = x_kind != "none"        # 반사도 로그 척도이므로 % 해석 대상이다
        B = fit.params[x]               # 비표준화 회귀계수(B, 로버스트 여부와 무관하게 동일)

        if hc3:
            # 로버스트(HC3) 표준오차에서 나온 t·유의확률로 유의성을 판정한다.
            i = fit.model.exog_names.index(x)     # 상수항을 포함한 전체 순서에서의 위치
            t = float(rob_tvalues[i])   # 로버스트 t (= B / 로버스트 SE)
            p = float(rob_pvalues[i])   # 로버스트 유의확률
        else:
            t = fit.tvalues[x]          # 일반 OLS t-통계량
            p = fit.pvalues[x]          # 일반 OLS 계수 유의확률

        # 5-2) 유의성·방향 판정
        # 유의성 판정 (유의수준 0.05 기준)
        if p < 0.05:    sig_word = "유의한"
        else:           sig_word = "유의하지 않은"

        # p값 APA 표기 (앞자리 0 생략)
        if p < 0.001:   p_text = "p < .001"
        else:           p_text = f"p = {p:.3f}".replace("0.", ".")

        # 계수 부호로 증가/감소 방향 결정
        # (문장의 주어가 반사값이면 이 방향은 반사값 기준의 방향이다)
        if B > 0:       direction = "증가"
        else:           direction = "감소"

        # 5-3) 변화 표현과 원 변수 기준 방향
        # 변환 종류별 독립변수 변화 표현 (% 해석의 기준이 되는 값이 무엇인지가 핵심이다)
        x_change = {
            "reflect": f"**(1+max-{x})가 1% 증가**할 때",
            "log1p":   f"**(1+{x})가 1% 증가**할 때",
            "log":     f"{x}가 **1% 증가**할 때",
            "none":    f"{x}가 **1 증가**할 때",
        }[x_kind]

        # 반사 변환은 대소 관계를 뒤집으므로, 반사한 변수가 홀수 개면 원 변수 기준 방향이 반대다.
        # (x·y 둘 다 반사면 두 번 뒤집혀 원래대로 돌아온다)
        reflected = (x_kind == "reflect") + (y_kind == "reflect")

        if not reflected:   note = ""   # 반사가 없으면 문장을 그대로 읽으면 된다
        else:
            if reflected % 2:   orig_direction = opposite[direction]
            else:               orig_direction = direction

            note = note_template.format(x=x, y=yname, orig_direction=orig_direction)

        # 5-4) 효과 크기 계산
        # 효과 크기: x·y가 각각 비율(%) 기준인지에 따라 값·단위가 정해진다
        if not x_pct and not y_pct:      # 원본 → 절대량 그대로
            mag, unit, approx = f"{abs(B):.2f}", "", ""
        elif x_pct and not y_pct:        # 독립변수만 로그 → 1% 증가당 절대 변화 ≈ B×ln(1.01)
            mag, unit, approx = f"{abs(B * np.log(1.01)):.3f}", "", "약 "
        elif not x_pct and y_pct:        # 종속변수만 로그 → (e^B − 1)×100 %
            mag, unit, approx = f"{abs((np.exp(B) - 1) * 100):.2f}", "%", "약 "
        else:                            # 둘 다 로그 → 탄력성 B %
            mag, unit, approx = f"{abs(B):.2f}", "%", "약 "

        effect = effect_template.format(
            x_change=x_change, y_target=y_target,
            approx=approx, mag=mag, unit=unit, direction=direction,
        )

        # 하나의 독립변수 → 하나의 불릿 문장
        lines.append(line_template.format(
            x=x, B=round(B, 2), y=yname, sig=sig_word,
            df=df_resid, t=round(t, 2), p=p_text, effect=effect, note=note,
        ))

    report = "\n".join(lines)   # 불릿 문장을 하나의 markdown 리스트로 합친다

    # --- 6) 로그·반사 변환 사용 시 주의 각주 ---
    uses_log1p = (y_kind == "log1p") or bool(log1p_x)
    if uses_log1p:
        report += (
            "\n\n> ※ **log1p**(=ln(1+·))의 % 해석은 변수 자체가 아니라 **(1+변수)** 기준이며, "
            "값이 클 때만 위 근사가 성립한다.      \n(0·작은 값 구간에서는 원본처럼 동작해 부정확)      \n"
            "이 구간에서는 부호·유의성 중심으로 해석하거나 예측값을 expm1로 원 척도에서 비교한다."
        )

    uses_reflect = (y_kind == "reflect") or bool(reflect_x)
    if uses_reflect:
        report += (
            "\n\n> ※ **반사 후 log1p**(=ln(1+max-·))는 값의 대소가 뒤집힌 변환이다. "
            "위 %·증감은 **(1+max-변수)** 기준이고,     \n"
            "원 변수 기준 방향은 각 문장 끝 괄호에 적었다.      \n"
            "변수가 **최댓값에 가까운 구간**에서는 위 근사가 부정확하므로 부호·유의성 중심으로 읽고,      \n"
            "원 척도 값은 **max-(exp(변환값)-1)** 로 되돌려 비교한다."
        )

    # --- 7) 로버스트 표준오차 사용 시 주의 각주 ---
    if hc3:
        report += (
            "\n\n> ※ 위 **t**와 **유의확률**은 등분산 가정이 충족되지 않은 경우를 대비해 "
            "**HC3 로버스트 표준오차**로 계산한 값이다.     \n"
            "회귀계수(B)와 효과 크기 해석은 일반 OLS와 동일하며,     \n"
            "표준오차만 이분산에 강건하게 보정되어 유의성 판정이 달라질 수 있다."
        )

    return report


def auto_ols(data, y, report=True,
             log_y=False, log_x=None, log1p_y=False, log1p_x=None,
             reflect_y=False, reflect_x=None, test=True, 
             plot=False, width=1280, height=640,
             backward=False, alpha=0.05):
    """회귀모델 적합부터 보고서 출력·가정 검정까지 한 번에 수행한다.

    Args:
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        y: 종속변수로 사용할 컬럼명.
        report (bool): 모형 적합도 보고서(회귀계수표·해설) 출력 여부 (기본값: True).
        log_y (bool): 종속변수에 log 변환을 적용했는지 여부 (기본값: False).
        log_x (list): log 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        log1p_y (bool): 종속변수에 log1p 변환을 적용했는지 여부 (기본값: False).
        log1p_x (list): log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        reflect_y (bool): 종속변수에 반사 후 log1p 변환을 적용했는지 여부 (기본값: False).
        reflect_x (list): 반사 후 log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        test (bool): 회귀모형 가정 검정 수행 여부 (기본값: True).
        plot (bool): 가정 검정 시 그래프를 함께 그릴지 여부 (기본값: False).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
        backward (bool): 후진소거법으로 유의하지 않은 독립변수를 제거할지 여부 (기본값: False).
        alpha (float): 후진소거법의 변수 제거 기준 유의수준 (기본값: 0.05).

    Returns:
        적합이 완료된 회귀분석 결과 객체. 등분산 위배 여부가 `use_hc3_`(bool) 속성으로 붙는다.
    """ 
    # 빈 줄 출력 (출력 결과의 여백을 위함)
    print()

    # # --- 1) 회귀모델 적합 ---
    # fit = fit_model(data, y)

    # # --- 2) 등분산성 가정 확인 ---
    # lm_stat, lm_p, f_stat, f_p = het_breuschpagan(fit.resid, fit.model.exog)
    # # 등분산 충족시 True, 위배시 False (유의수준 0.05 기준)
    # homoscedasticity = bool(float(f_p) >= 0.05)

    while True:
        # --- 1) 회귀모델 적합 ---
        fit = fit_model(data, y)

        # --- 2) 등분산성 가정 확인 ---
        lm_stat, lm_p, f_stat, f_p = het_breuschpagan(fit.resid, fit.model.exog)
        # 등분산 충족시 True, 위배시 False (유의수준 0.05 기준)
        homoscedasticity = bool(float(f_p) >= 0.05)

        if not backward:
            break   # 후진소거법이 아니면 반복문 종료

        report_vars = report_variables(fit, data, hc3=not homoscedasticity)
        # 등분산이면 일반 OLS의 유의확률, 위배되면 HC3 유의확률을 제거 기준으로 삼는다
        pvalues = report_vars["유의확률"] if homoscedasticity else report_vars["유의확률(HC3)"]

        # 독립변수가 하나뿐이거나 모두 유의하면 종료
        if len(pvalues) <= 1 or pvalues.max() < alpha:
            break

        # 유의확률이 가장 큰(=가장 유의하지 않은) 독립변수를 하나만 제거한다.
        # 여러 개를 한꺼번에 지우면, 변수 간 상관 때문에 원래는 유의해졌을 변수까지 사라진다.
        worst = report_vars.loc[pvalues.idxmax(), "독립변수"]
        print(f"유의하지 않은 독립변수 제거 → {worst} (p = {pvalues.max():.4f})")
        data = data.drop(columns=[worst])

    # 등분산 위배 여부를 결과 객체에 붙여 둔다.
    # 이미 위에서 판단한 값이므로, 보고 함수의 hc3 인자에 그대로 넘겨 쓰면
    # 같은 검정을 밖에서 다시 할 필요가 없다
    fit.use_hc3_ = not homoscedasticity


    # --- 3) 모형 적합도 출력 ---
    if report:
        display(Markdown("#### ▶︎ 모형 적합도"))
        # 회귀계수 보고 표(hc3는 등분산 충족 아닐 시 True로 설정)
        display(report_variables(fit, data, hc3=not homoscedasticity))
        display(Markdown(report_fitness(fit, log_y=log_y, log_x=log_x,
                                        log1p_y=log1p_y, log1p_x=log1p_x,
                                        reflect_y=reflect_y, reflect_x=reflect_x)))

    # --- 4) 회귀모형 가정 검정 ---
    # 보고서와 가정 검정이 모두 출력되는 경우, 구분을 위해 수평선 추가
    if report and test:
        display(Markdown("---"))

    # 회귀모형 가정 검정 (선형성 → 정규성 → 등분산성 → 독립성)
    if test:
        display(Markdown("#### ▶︎ 회귀모형 가정 검정"))
        display(Markdown("##### 1) 선형성 검정"))
        test_linear(fit, plot=plot, width=width, height=height)
        display(Markdown("##### 2) 정규성 검정"))
        test_normal(fit, plot=plot, width=width, height=height)
        display(Markdown("##### 3) 등분산성 검정"))
        test_equalvar(fit)
        display(Markdown("##### 4) 독립성 검정"))
        test_independent(fit)

    # --- 5) 최종 적합 모델 객체 반환 ---
    return fit



def plot_beta(fit, data, palette=None, title=None, xlabel=None, ylabel=None,
              width=1280, height=None, save_path=None):
    """표준화 회귀계수(β)를 가로 막대그래프로 시각화해 독립변수의 영향력 순위를 보여준다.

    |β| 내림차순으로 배치하며, β의 순위는 영향력의 순위일 뿐 절대적 크기는 아니다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        palette (dict): 부호별 막대 색상. None이면 {'+': 파랑, '-': 빨강} (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 레이블 (기본값: None → "표준화 계수(β)").
        ylabel (str): y축 레이블 (기본값: None → "독립변수").
        width (int): 캔버스 가로 픽셀 (기본값: 1280).
        height (int): 캔버스 세로 픽셀. None이면 독립변수 수 × 80으로 자동 계산 (기본값: None).
        save_path (str): 이미지 저장 경로 (기본값: None).
    """
    # --- 1) 시각화용 데이터 전처리 ---
    # 회귀계수 표 리턴받기 - 베타값 자체는 hc3 여부와 무관하므로 hc3=False로 호출한다.
    vdf = report_variables(fit, data, hc3=False)
    rdf = vdf[["독립변수", "β"]].copy()
    rdf["부호"] = np.where(rdf["β"] > 0, "+", "-")   # 계수 부호(색상 구분용)

    # 독립변수가 많을수록 막대가 촘촘해지므로, 변수 하나당 80px씩 세로 공간을 확보한다
    if height is None:
        height = len(rdf) * 80

    # 부호별 기본 색상: 양(+)은 파랑, 음(-)은 빨강
    if palette is None:
        palette = {"+": "#0066ff", "-": "#ff3333"}

    # --- 2) 그래프 초기화 ---
    fig, ax = my_plot.init(width=width, height=height, title=title,
                            xlabel=xlabel if xlabel else "표준화 계수(β)",
                            ylabel=ylabel if ylabel else "독립변수")

    # --- 3) 가로 막대그래프 (값 축을 x로 두면 가로형이 된다) ---
    my_plot.barplot(rdf, x="β", y="독립변수", hue="부호", palette=palette, ax=ax)

    # --- 4) 막대 끝에 β 값 표기 ---
    # 양수 막대는 오른쪽 끝의 바깥쪽(ha="left"), 음수 막대는 왼쪽 끝의 바깥쪽(ha="right")에
    # 붙도록 정렬 기준을 뒤집고, 막대와 겹치지 않게 부호 방향으로 살짝 띄운다.
    for i in rdf.index:
        beta = rdf.loc[i, "β"]
        ax.text(x=beta + 0.001 * np.sign(beta), y=i, s=f"{beta:.2f}",
                va="center", ha="left" if beta > 0 else "right", color="black")

    # --- 5) 그래프 표시 (외부 ax를 받은 경우 표시는 호출자에게 맡긴다) ---
    my_plot.show(save_path=save_path)

# =====================================================================
# 전처리 파이프라인 — 플래그대로 전처리한 뒤 회귀모델을 적합한다
# =====================================================================
def fit_pipeline(data, y, nominal_cols=None, *,
                 # --- 1) 명목형 라벨링 (문자열 -> 정수) ---
                 labeling=True,             # 명목형 라벨링 수행 여부
                 # --- 2) 더미변수 인코딩 ---
                 encode=True,               # 더미변수 인코딩 수행 여부
                 # --- 3) 로그 변환 (대상은 왜도·첨도로 자동 선정) ---
                 log=False,                 # 로그 변환 수행 여부
                 # --- 4) 이상치 대체 (IQR 경계값, 행 삭제 없음) ---
                 outlier=False,             # 이상치 대체 수행 여부
                 # --- 5) 다중공선성 제거 (VIF) ---
                 vif=False,                 # 다중공선성 제거 수행 여부
                 vif_threshold=10.0,        # VIF 임계값
                 # --- 6) 정규화 ---
                 scale=False,               # 정규화 수행 여부
                 scale_method='standard',   # 사용할 스케일러 이름 (standard / minmax / robust)
                 # --- 7) 모델 적합 ---
                 backward=True,             # 후진소거법 수행 여부
                 alpha=0.05,                # 후진소거법의 변수 제거 기준 유의수준
                 # --- 기타 ---
                 name=None,                 # 모델을 구분할 이름. 결과 객체의 `name_` 속성이 된다
                 save_path=None,            # 전처리 완료 데이터의 저장 경로 (.xlsx/.xls/.csv)
                 verbose=True):             # 단계별 전처리 내역 출력 여부
    """플래그로 지정한 전처리를 수행한 뒤 회귀모델을 적합한다. 결측치는 없다고 전제한다.

    Args:
        data (DataFrame): 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        y (str): 종속변수로 사용할 컬럼명.
        nominal_cols (list): 명목형 컬럼명 리스트. None이면 타입 자동 선택 (기본값: None).
        labeling (bool): 명목형 라벨링(문자열 -> 정수) 수행 여부 (기본값: True).
        encode (bool): 더미변수 인코딩 수행 여부 (기본값: True).
        log (bool): 로그 변환 수행 여부. 대상은 왜도·첨도로 자동 선정한다 (기본값: False).
        outlier (bool): 이상치를 IQR 경계값으로 대체할지 여부 (기본값: False).
        vif (bool): 다중공선성 제거 수행 여부 (기본값: False).
        vif_threshold (float): VIF 임계값 (기본값: 10.0).
        scale (bool): 정규화 수행 여부 (기본값: False).
        scale_method (str): 사용할 스케일러 이름 (기본값: 'standard').
        backward (bool): 후진소거법 수행 여부 (기본값: True).
        alpha (float): 후진소거법의 변수 제거 기준 유의수준 (기본값: 0.05).
        name (str): 모델을 구분할 이름. 결과 객체의 `name_` 속성이 된다 (기본값: None).
        save_path (str): 전처리 완료 데이터의 저장 경로 (.xlsx/.xls/.csv) (기본값: None).
        verbose (bool): 단계별 전처리 내역 출력 여부 (기본값: True).

    Returns:
        적합이 완료된 회귀분석 결과 객체. 보고에 필요한 정보가 아래 속성으로 함께 붙는다.
            - `log_y_` (bool) / `log_x_` (list): 순수 log 변환 정보
            - `log1p_y_` (bool) / `log1p_x_` (list): log1p 변환 정보
            - `reflect_y_` (bool) / `reflect_x_` (list): 반사 후 로그 변환 정보
            - `reflect_y_max_` (float): 종속변수를 반사할 때 쓴 최댓값 (역변환용)
            - `data_` (DataFrame): 전처리가 끝난 데이터 (β·VIF 계산용)
            - `use_hc3_` (bool): 등분산 위배 여부 (`auto_ols`가 붙인다)
    """
    # --- 1) 종속변수 확인 및 작업본 준비 ---
    if y not in data.columns:
        raise KeyError(f"종속변수 '{y}'가 데이터프레임의 컬럼에 존재하지 않습니다.")

    df = data.copy()    # 원본을 보존하기 위해 복사본으로 작업

    # --- 2) 명목형 컬럼 확정 ---
    # 지정이 없으면 category/object 타입을 자동으로 선택한다
    if nominal_cols is None:
        nominal_cols = list(df.select_dtypes(include=['category', 'object']).columns)
    else:
        missing = []
        for c in nominal_cols:
            if c not in df.columns:
                missing.append(c)

        if missing:
            raise KeyError(f'df 에 존재하지 않는 컬럼입니다: {missing}')

    # 종속변수는 명목형 목록에서 제외한다 (회귀의 종속변수는 연속형이어야 한다)
    nominals = []
    for c in nominal_cols:
        if c != y:
            nominals.append(c)

    nominal_cols = nominals

    # --- 3) 연속형 독립변수 확정 ---
    # 수치형 중에서 종속변수와 명목형을 뺀 나머지.
    # 로그변환·이상치대체·다중공선성·정규화의 대상이 되며, 단계마다 갱신된다
    continuous = []
    for c in df.select_dtypes(include='number').columns:
        if c != y and c not in nominal_cols:
            continuous.append(c)

    # --- 4) 변환 정보 초기화 ---
    # 로그 변환 정보 (반환되는 fit 객체에 붙여 계수 해석에 사용한다).
    # 순수 log 와 log1p 는 % 해석의 기준이 다르므로(전자는 변수, 후자는 1+변수) 따로 관리한다
    log_y = False
    log_x = []
    log1p_y = False
    log1p_x = []

    # 반사(좌측 꼬리) 변환 정보. 해석의 기준·방향이 log1p 와 다르므로 따로 관리한다.
    # 종속변수를 반사한 경우 원 척도로 되돌리려면 변환 당시의 최댓값이 반드시 필요하다
    reflect_y = False
    reflect_x = []
    reflect_y_max = None

    # --- 5) 대상 요약 출력 ---
    if verbose:
        print(f'대상: {df.shape[0]}행 x {df.shape[1]}열 | 종속변수: {y}')
        print(f'명목형: {nominal_cols}')
        print(f'연속형: {continuous}')

    # --- 6) 명목형 라벨링 ---
    if labeling and nominal_cols:
        if verbose:
            print('\n명목형 라벨링')

        df = my_prep.labeling(df, columns=nominal_cols, verbose=verbose)

    # --- 7) 더미변수 인코딩 ---
    if encode and nominal_cols:
        if verbose:
            print('\n더미변수 인코딩')

        df = my_prep.dummies(df, columns=nominal_cols, drop_first=True, verbose=verbose)

    # --- 8) 로그 변환 ---
    if log:
        # 8-1) 변환 후보 추리기
        # 연속형 독립변수와 종속변수가 후보다 (실제로 무엇을 변환할지는 통계량이 정한다)
        scope = list(continuous) + [y]

        # 값이 두 종류뿐인 이진 변수(0/1 플래그 등)는 후보에서 뺀다.
        # '1% 증가' 라는 해석 자체가 성립하지 않고, 로그를 씌워도 분포가 대칭이 되지 않는다
        binary_cols = []
        targets = []

        for c in scope:
            if df[c].dropna().nunique() <= 2:    binary_cols.append(c)
            else:                                targets.append(c)

        scope = targets

        # 8-2) 왜도·첨도로 변환 대상 자동 선정
        # 우측 꼬리는 값의 위치에 따라 log 와 log1p 로 갈리고, 좌측 꼬리는 반사 후 log1p로 구분
        log_columns = []
        log1p_columns = []
        reflect_columns = []

        if scope:
            desc = my_qtcheck.numerical_summary(df, columns=scope)
            log_columns = desc.index[desc['log_need'] == 'log'].tolist()
            log1p_columns = desc.index[desc['log_need'] == 'log1p'].tolist()
            reflect_columns = desc.index[desc['log_need'] == 'reverse_log1p'].tolist()

        if verbose:
            print('\n로그 변환')

            if binary_cols:
                print(f'이진 변수는 자동 선정에서 제외: {binary_cols}')

        # 8-3) 변환 실행 (종속변수를 반사하면 역변환용 최댓값을 먼저 남겨 둔다)
        if y in reflect_columns:
            reflect_y_max = float(df[y].max())

        df = my_prep.log_transform(df, log_columns=log_columns,
                                   log1p_columns=log1p_columns,
                                   reflect_columns=reflect_columns, verbose=verbose)

        # 8-4) 계수 해석에 쓸 변환 정보 기록
        # 세 변환은 % 해석의 기준이 서로 다르므로(변수 / 1+변수 / 1+max-변수) 목록을 섞지 않는다
        log_y = y in log_columns
        log1p_y = y in log1p_columns
        reflect_y = y in reflect_columns

        for column_list, name_list in ((log_columns, log_x),
                                       (log1p_columns, log1p_x),
                                       (reflect_columns, reflect_x)):
            for c in column_list:
                if c != y:
                    name_list.append(c)

    # --- 9) 이상치 대체 ---
    # 연속형 독립변수만 대상으로 한다 (종속변수를 자르면 예측 대상 자체가 왜곡된다)
    if outlier and continuous:
        if verbose:
            print('\n이상치 대체')

        df = my_prep.replace_outlier(df, columns=continuous, verbose=verbose)

    # --- 10) 다중공선성 제거 ---
    if vif and continuous:
        if verbose:
            print(f'\n다중공선성 제거 (VIF >= {vif_threshold})')

        df = my_prep.reduce_vif(df, columns=continuous,
                                threshold=vif_threshold, verbose=verbose)

        # 제거된 변수를 반영해야 이후 정규화 단계에서 없는 컬럼을 찾지 않는다
        survived = []
        for c in continuous:
            if c in df.columns:
                survived.append(c)

        continuous = survived

    # --- 11) 정규화 ---
    if scale and continuous:
        if verbose:
            print('\n정규화')

        df = my_prep.scaling(df, columns=continuous,
                             method=scale_method, verbose=verbose)

    # --- 12) 전처리 완료 데이터 저장 (선택) ---
    if save_path:
        # 저장 폴더 준비 (경로에 없는 폴더가 있으면 만들어 준다)
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        # 확장자에 추출
        ext = os.path.splitext(save_path)[1].lower()

        # 확장자에 따른 데이터 저장
        if ext in ('.xlsx', '.xls'):    df.to_excel(save_path, index=False)
        elif ext == '.csv':             df.to_csv(save_path, index=False, encoding='utf-8-sig')
        else:                           raise ValueError(f"{ext}(은)는 지원하지 않는 저장 형식입니다")

        if verbose:
            print(f'\n전처리 데이터 저장: {save_path} '
                  f'({df.shape[0]}행 x {df.shape[1]}열)')

    # --- 13) 모델 적합 전 데이터에 이상이 없는지 판단 ---
    # 13-1) 숫자로 바뀌지 않은 컬럼 확인 (남아 있으면 OLS 가 알 수 없는 오류를 낸다)
    remain = []
    for c in df.columns:
        if c not in df.select_dtypes(include='number').columns:
            remain.append(c)

    if remain:
        raise ValueError(f'숫자로 변환되지 않은 컬럼이 남아 있습니다: {remain}\n'
                         f'labeling=True 또는 encode=True 로 설정하세요.')

    # 13-2) 결측치 확인 (남아 있으면 OLS 가 MissingDataError 를 낸다)
    na_cols = df.columns[df.isna().any()].tolist()

    if na_cols:
        raise ValueError(f'결측치가 있는 컬럼이 있습니다: {na_cols}\n'
                         f'데이터 품질 점검 단계에서 먼저 처리하세요.')

    # --- 14) 모델 적합 ---
    fit = auto_ols(df, y, backward=backward, alpha=alpha,
                   log_y=log_y, log_x=log_x,
                   log1p_y=log1p_y, log1p_x=log1p_x,
                   reflect_y=reflect_y, reflect_x=reflect_x,
                   report=False, test=False)

    # --- 15) 보고에 필요한 정보를 결과 객체에 붙여 반환 ---
    # 로그 변환 정보는 report_fitness(), report_variables_text() 에 그대로 넘겨 쓴다
    fit.log_y_ = log_y
    fit.log_x_ = log_x
    fit.log1p_y_ = log1p_y
    fit.log1p_x_ = log1p_x

    # 반사 변환 정보. 최댓값은 compare_models() 가 예측값을 원 척도로 되돌릴 때 쓴다
    fit.reflect_y_ = reflect_y
    fit.reflect_x_ = reflect_x
    fit.reflect_y_max_ = reflect_y_max

    # 전처리가 끝난 데이터. report_variables(), plot_beta() 가 β·VIF 계산에 사용해야 한다.
    fit.data_ = df

    # 모델을 구분할 이름 (compare_models 가 딕셔너리 키로 채워 주기도 한다)
    fit.name_ = name

    return fit


# =====================================================================
# 모델 성능 비교 — 여러 모델의 지표를 한 표로 모아 성능순으로 정렬한다
# =====================================================================
def compare_models(fits, metric='RMSE', sub_metric='변수수', tolerance=0.05,
                   digits=4, report=True):
    """여러 회귀모델의 성능지표를 한 표로 정리해 성능이 좋은 순으로 정렬하고, 최고 모델을 반환한다.
    주 지표 1위와의 격차가 tolerance 이내면 '근소 격차 그룹'으로 묶어 보조 지표로 순서를 정한다.
    종속변수를 log1p·반사 변환한 모델은 예측값을 원본 척도로 되돌려 RMSE·MAE·R²를 계산한다.

    Args:
        fits (dict): {모델이름: 적합된 회귀분석 결과 객체} 형태의 딕셔너리.
        metric (str): 정렬 기준이 되는 주 성능평가지표 (기본값: 'RMSE').
        sub_metric (str): 근소 격차 그룹 안에서 적용할 보조 지표. None이면 미사용 (기본값: '변수수').
        tolerance (float): 근소 격차로 판단할 주 지표의 상대격차. 0이면 순수 크기 비교 (기본값: 0.05).
        digits (int): 표에 표시할 소수점 자릿수 (기본값: 4).
        report (bool): 성능 비교표를 화면에 출력할지 여부 (기본값: True).

    Returns:
        성능이 가장 좋은 모델의 회귀분석 결과 객체(표의 첫 행). 아래 속성이 함께 붙는다.
            - `name_` (str): 모델 이름. `fits` 의 키에서 채워진다
            - `score_table_` (DataFrame): 모델명을 인덱스로 하는 성능 비교표.
              성능이 좋은 모델이 위에 오며, 맨 끝에 1위 대비 상대격차인 `Gap(%)` 컬럼이 붙는다

    Raises:
        TypeError: `fits` 가 딕셔너리가 아니거나 값이 회귀분석 결과 객체가 아닌 경우.
        ValueError: `fits` 가 비었거나 지표 이름·tolerance 가 유효하지 않은 경우.
    """
    # --- 1) 지표별 '성능이 좋은 방향' 정의 (True = 값이 클수록 좋음) ---
    metrics = {
        '변수수': False,          # 같은 성능이라면 변수가 적은 모델이 간명하다
        'R2(모델척도)': True,
        'Adj.R2': True,
        'AIC': False,
        'BIC': False,
        'R2(원본척도)': True,
        'RMSE': False,
        'MAE': False,
    }

    # 종속변수의 척도가 다르면 비교할 수 없는 지표
    # (모델에 들어간 값 그대로 계산되므로 로그 변환 여부가 다르면 값의 단위 자체가 다르다)
    scale_sensitive = ['R2(모델척도)', 'Adj.R2', 'AIC', 'BIC']

    # --- 2) 입력 검증 ---
    if not isinstance(fits, dict):
        raise TypeError(f'fits 는 딕셔너리여야 합니다: {type(fits).__name__}')

    if not fits:
        raise ValueError('비교할 모델이 없습니다.')

    for name, fit in fits.items():
        if not hasattr(fit, 'fittedvalues'):
            raise TypeError(f"'{name}' 의 값이 회귀분석 결과 객체가 아닙니다: "
                            f'{type(fit).__name__}')

    for m in [metric, sub_metric]:
        if m is not None and m not in metrics:
            raise ValueError(f"지원하지 않는 지표입니다: '{m}' "
                             f'(사용 가능: {list(metrics.keys())})')

    if tolerance < 0:
        raise ValueError(f'tolerance 는 0 이상이어야 합니다: {tolerance}')

    # --- 3) 모델별 성능지표 계산 ---
    result = []
    log_flags = []    # 종속변수의 척도가 섞여 있는지 확인하기 위해 기록

    for name, fit in fits.items():
        # --- 3-1) 필요한 정보 추출
        # fit_pipeline() 이 붙여 둔 로그 변환 정보 추출. 없으면 변환하지 않은 것으로 본다
        # -> getattr() 로 속성이 없으면 False/None 을 반환하도록 한다
        plain_log_y = getattr(fit, 'log_y_', False)     # 순수 log
        log1p_y = getattr(fit, 'log1p_y_', False)       # log1p
        # 종속변수를 반사 변환(좌측 꼬리)한 경우와 그때 사용한 최댓값
        reflect_y = getattr(fit, 'reflect_y_', False)
        reflect_max = getattr(fit, 'reflect_y_max_', None)

        # 종속변수의 척도가 바뀐 모델인지 기록 (셋 중 무엇이든 원 척도가 아니다)
        log_flags.append(bool(plain_log_y or log1p_y or reflect_y))

        # 실제값과 예측값을 원본 척도로 되돌린다.
        # 이렇게 해야 로그 변환 여부가 다른 모델끼리도 오차를 비교할 수 있다
        y_true = fit.model.endog
        y_pred = fit.fittedvalues

        if reflect_y and reflect_max is None:
            # 최댓값을 모르면 되돌릴 수 없으므로, 에러처리
            raise ValueError(f"'{name}' 은 종속변수를 반사 변환했지만 역변환에 필요한 "
                             f'최댓값(reflect_y_max_)이 없습니다.')

        # --- 3-2) 원 척도로 되돌리기 (로그 변환 여부에 따라 다르다) ---
        if plain_log_y:
            # log(y) 의 역변환: exp(y)
            y_true, y_pred = np.exp(y_true), np.exp(y_pred)
        elif log1p_y:
            y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)
        elif reflect_y:
            # 반사 변환의 역변환: max - (exp(y)-1)
            y_true = reflect_max - np.expm1(y_true)
            y_pred = reflect_max - np.expm1(y_pred)

        # --- 3-3) 성능지표 계산 ---
        result.append({
            '모델': name,
            '변수수': int(fit.df_model),        # 상수항을 제외한 독립변수 개수
            'R2(모델척도)': fit.rsquared,       # 종속변수 척도가 같을 때만 비교 가능
            'Adj.R2': fit.rsquared_adj,
            'AIC': fit.aic,
            'BIC': fit.bic,
            'R2(원본척도)': r2_score(y_true, y_pred),
            'RMSE': root_mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
        })

    rdf = DataFrame(result).set_index('모델')

    # --- 4) 종속변수의 척도가 섞였는데 척도 의존 지표로 정렬하려는 경우 경고 ---
    if len(set(log_flags)) > 1 and metric in scale_sensitive:
        applied = sum(log_flags)
        print(f'⚠ 종속변수의 척도가 서로 다른 모델이 섞여 있습니다'
              f'(로그·반사 변환 적용: {applied}개 / 미적용: {len(log_flags) - applied}개).\n'
              f"  '{metric}' 지표는 같은 척도끼리만 비교할 수 있습니다. "
              f"'RMSE' 또는 'MAE' 를 사용하세요.")

    # --- 5) 1위 대비 주 지표의 상대격차 계산 ---
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

    rdf['Gap(%)'] = (diff / denominator * 100).round(2)    # 양수일수록 1위보다 나쁨

    # --- 6) 근소 격차 그룹을 먼저 정렬하고 나머지를 뒤에 붙인다 ---
    # 주 지표가 사실상 비슷한(격차가 tolerance 이내인) 모델끼리는 보조 지표로 순서를 정한다
    close = rdf['Gap(%)'] <= tolerance * 100

    by = [metric]
    ascending = [not higher_is_better]

    if sub_metric:
        # 보조 지표를 앞에 두어야 근소 격차 그룹 안에서 우선 적용된다
        by.insert(0, sub_metric)
        ascending.insert(0, not metrics[sub_metric])

    front = rdf[close].sort_values(by=by, ascending=ascending)
    back = rdf[~close].sort_values(by=[metric], ascending=[not higher_is_better])

    score_table = concat([front, back]).round(digits)

    # --- 7) 성능표 출력 ---
    if report:
        display(score_table)

    # 각 모델에 딕셔너리 키를 이름으로 새겨 둔다 (직접 지정한 name_ 이 없을 때만)
    for model_name, fit in fits.items():
        if getattr(fit, 'name_', None) is None:
            fit.name_ = model_name

    # --- 8) 최고 성능 모델을 반환한다 ---
    # 표는 성능순으로 정렬되어 있으므로 첫 행이 곧 최고 모델이다
    best = fits[score_table.index[0]]
    best.score_table_ = score_table
    
    return best


# =====================================================================
# 최종 모델 종합 보고 — 성능 보고와 가정 검정을 한 번에 수행한다
# =====================================================================
def report_model(fit, title=True, plot=True):
    """적합된 회귀모델의 성능 보고와 가정 검정을 한 번에 출력한다.

    `fit_pipeline`·`auto_ols` 가 결과 객체에 붙여 둔 정보(`log1p_y_`·`log1p_x_`·
    `use_hc3_`·`data_`)를 사용하므로, 이 값들을 따로 준비해 넘길 필요가 없다.

    출력 구성 (마크다운 제목 포함):
        ### ▶︎ 성능 보고
            #### 1) 모형 적합도  2) 회귀계수 보고표  3) 영향력 순위 시각화  4) 회귀계수 해석 문장
        ---
        ### ▶︎ 회귀모형 가정 검정
            #### 1) 선형성  2) 정규성  3) 등분산성  4) 독립성

    Args:
        fit: `fit_pipeline` 또는 `auto_ols` 로 적합된 회귀분석 결과 객체.
            `log1p_y_`·`log1p_x_`·`use_hc3_`·`data_` 속성이 붙어 있어야 한다.
        title (bool): 모델 이름(`name_`)이 있으면 맨 위에 2수준 제목으로 출력할지 여부 (기본값: True).

    Raises:
        AttributeError: 보고에 필요한 속성이 결과 객체에 없는 경우.
    """
    # --- 1) 필요한 속성 확인 (fit_pipeline/auto_ols 산출물이 아니면 안내) ---
    need = ['log1p_y_', 'log1p_x_', 'use_hc3_', 'data_']
    missing = []
    for attr in need:
        if not hasattr(fit, attr):
            missing.append(attr)

    if missing:
        raise AttributeError(
            f'보고에 필요한 속성이 없습니다: {missing}\n'
            f'report_model 은 fit_pipeline() 또는 auto_ols() 로 적합한 모델에 사용하세요.')

    data = fit.data_
    log1p_y = fit.log1p_y_
    log1p_x = fit.log1p_x_
    hc3 = fit.use_hc3_

    # 순수 log 와 반사 변환 정보. 이 속성이 없는 예전 결과 객체도 그대로 동작하도록 기본값을 둔다
    log_y = getattr(fit, 'log_y_', False)
    log_x = getattr(fit, 'log_x_', [])
    reflect_y = getattr(fit, 'reflect_y_', False)
    reflect_x = getattr(fit, 'reflect_x_', [])

    # --- 2) 모델 이름 제목 (선택) ---
    # 제목은 수준에 상관없이 앞에 빈 줄을 하나 두어 위 내용과 간격을 준다
    def heading(text):
        print()
        display(Markdown(text))

    if title and getattr(fit, 'name_', None) is not None:
        heading(f"## 최종 모델: {fit.name_}")

    # --- 3) 성능 보고 ---
    heading("### ▶︎ 성능 보고")

    heading("#### 1) 모형 적합도")
    display(Markdown(report_fitness(fit, log_y=log_y, log_x=log_x,
                                    log1p_y=log1p_y, log1p_x=log1p_x,
                                    reflect_y=reflect_y, reflect_x=reflect_x)))

    heading("#### 2) 회귀계수 보고표")
    display(report_variables(fit, data, hc3=hc3))

    heading("#### 3) 영향력 순위 시각화 (표준화 계수 β)")
    plot_beta(fit, data, title="최종 모델의 표준화 회귀계수(β) — 영향력 순위")

    heading("#### 4) 회귀계수 해석 문장")
    display(Markdown(report_variables_text(fit, log_y=log_y, log_x=log_x,
                                           log1p_y=log1p_y, log1p_x=log1p_x,
                                           reflect_y=reflect_y, reflect_x=reflect_x, hc3=hc3)))

    # --- 4) 성능 보고와 가정 검정 사이 구분선 ---
    display(Markdown("---"))

    # --- 5) 회귀모형 가정 검정 ---
    heading("### ▶︎ 회귀모형 가정 검정")

    heading("#### 1) 선형성 검정")
    test_linear(fit, plot=plot, title="적합값 대비 잔차 (lowess 추세선)")

    heading("#### 2) 정규성 검정")
    test_normal(fit, plot=plot)

    heading("#### 3) 등분산성 검정")
    test_equalvar(fit)

    heading("#### 4) 독립성 검정")
    test_independent(fit)
