import numpy as np
import seaborn as sb
from IPython.display import display, Markdown
from pandas import DataFrame
from statsmodels.api import add_constant, OLS
from statsmodels.stats.diagnostic import linear_reset
from scipy.stats import zscore, probplot, shapiro, kstest
from statsmodels.stats.api import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from . import my_plot, my_stats

def fit_model(data:DataFrame, y:str, summary=False):
    """
    statsmodels의 OLS를 이용해 선형회귀 모델을 적합한다.

    종속변수 'y'를 제외한 나머지 모든 컬럼을 독립변수로 사용하며,
    절편(상수항)을 자동으로 추가한 뒤 최소자승법으로 회귀계수를 추정한다.

    Args:
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임
        y: 종속변수로 사용할 컬럼명. 'data'에 반드시 존재해야함
        summary: True로 설정하면 적합된 모델의 요약 통계량을 출력함, Defaults to False
    
    Returns:
        적합이 완료된 회귀분석 결과 객체
    """
    if y not in data.columns:
        raise KeyError(f'종속변수 "{y}"가 데이터프레임의 컬럼에 존재하지 않습니다.')
    
    # 종속변수(y_series)와 독립변수(x_input)를 분리
    x = data.drop(columns=[y])
    y_series = data[y]

    # 독립변수에 절편(상수항) 추가
    x_input = add_constant(x)

    # OLS 모델 객체 생성
    model = OLS(y_series, x_input)

    # 모델 적합(fit)
    fit = model.fit()

    # 적합된 모델의 요약 통계량 출력 여부 확인
    if summary:
        print(fit.summary())

    # 적합된 모델 객체(분석 결과) 반환
    return fit

def predict(fit, new_data) -> DataFrame:
    """
    적합된 회귀모델을 이용해 새로운 데이터에 대한 예측값을 계산한다.

    Args:
        fit: 'fit_model' 함수로 적합된 회귀분석 결과 객체
        new_data: 예측에 사용할 새로운 데이터프레임. 독립변수 컬럼만 포함해야 한다.

    Returns:
        DataFrame: 새로운 데이터에 대한 예측값을 포함하는 데이터프레임. 컬럼명은 'predicted'로 설정된다.
    """
    # 새로운 데이터에 절편(상수항) 추가
    new_data_with_const = add_constant(new_data, has_constant='add')

    # 예측값 계산
    predictions = fit.predict(new_data_with_const)

    # 예측값을 DataFrame으로 반환
    return DataFrame(predictions, columns=['pred'])

#===============================
# 선형성 가정 검정 함수
#===============================
def test_linear(fit, alpha=0.05, plot=True, palette=None, title=None, 
                xlabel=None, ylabel=None, width=1280, height=640, save_path=None):
    """
    잔차의 선형성(모형 설정 오류)을 검정한다.
    Ramsey RESET Test(power=2)를 수행하여 적합된 선형모형에 고차항을 추가했을 때 유의미한 설명력이 남는지를 확인한다.
    고차항이 유의하면(=p < alpha) 직선으로는 잡아내지 못한 곡선 관계가 남아있다는 뜻이므로 선형성 가정이 위배된다.

    Args:
        fit: 'fit_modal' 함수로 적합된 회귀분석 결과 객체 
        alpha (float): 유의수준(기본값: 0.05) 
        plot (bool) : 적합값-잔차 산점도(lowess 추세선 포함)를 시각화할지 여부(기본값 : True) 
        palette (str) : 산점도 점 색상에 팔레트 이름. None이면 기본색 (기본값 : None) 
        title (str) : 그래프 제목 (기본값 : None) 
        xlabel (str) : x축 라벨 (기본값: None → "적합값(예측값)") 
        ylabel (str) : y축 라벨 (기본값: None → "잔차(residual)")
        width (int) : 그래프 너비 (기본값 : 1280) 
        height (int) : 그래프 높이 (기본값: 640) 
        save_path (str) : 그래프 저장경로 (기본값 : None)
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
    result_df = DataFrame([{
        "statistic": round(fvalue,4),
        "p-value":round(pvalue,4),
        "linearity":linearity,
        "result":conclusion
    }], index=["Ramsey RESET"])

    display(result_df)  # 결과표 출력

    # --- 4) 시각화: 적합값 대비 잔차 산점도 + lowess 추세선
    if plot:
        # 팔레트가 지정되면 첫번째 색을 산점도 점 색상으로 사용
        point_color = sb.color_palette(palette)[0] if palette else '#328cc1'

        plot_df = DataFrame({'y_pred':fit.fittedvalues, 'resid':fit.resid})

        fig, ax = my_plot.init(width=width, height=height, title=title, xlabel=xlabel if xlabel else "적합값(예측값)",
                               ylabel=ylabel if ylabel else "잔차(residual)")
        
        # 잔차 = 0 기준선(파란 점선)
        my_plot.lineplot(x=[plot_df['y_pred'].min(), plot_df['y_pred'].max()],
                         y=[0,0], color='blue', linestyle='--', ax=ax)
        
        # 잔차 산점도 + lowess(비선형) 추세선(빨강)
        sb.regplot(data=plot_df, x='y_pred', y='resid', lowess=True, 
                   scatter_kws={'color':point_color, 'edgecolor':'#ffffff', 'alpha':0.8},
                   line_kws={'color':'red'}, ax=ax)
        
        my_plot.show(save_path=save_path)



#===============================
# 정규성 가정 검정 함수
#===============================
def test_normal(fit, alpha=0.05, plot=True, palett=None, width=1280, height=640):
    """
    잔차의 정규성을 두 가지 방법으로 검정하고 진단 결과를 순서대로 출력한다.

     Args:
        fit: 'fit_modal' 함수로 적합된 회귀분석 결과 객체 
        alpha (float): 유의수준(기본값: 0.05) 
        plot (bool) : Q-Q 플롯과 √MSE 잔차도를 함께 그릴지 여부 (기본값: True)
        palette (str) : 산점도 점 색상에 팔레트 이름. None이면 기본색 (기본값 : None) 
        title (str) : 그래프 제목 (기본값 : None) 
        width (int) : 그래프 너비 (기본값 : 1280) 
        height (int) : 그래프 높이 (기본값: 640) 
    """
    ### --- 1) 잔차 추출 및 표본수에 따른 검정 선택 ---
    resid =fit.resid                                # 잔차(residual) 추출
    n = len(resid)                                  # 표본수(n) 확인

    if n < 30:
        method = "Shapiro-Wilk"                     # 표본수가 30 미만이면 Shapiro-Wilk 검정 사용
        s, p = shapiro(resid)                       # Shapiro-Wilk 검정 통계량 p값
    else:
        method = "Kolmogorov-Smirnov"               # 표본수가 30 이상이면 Kolmogorov-Smirnov 검정 사용
        # 표본 평균, 표준편차로 표준화한 뒤 표준정규준포(N(0,1))와 비교
        # kstest에 loc/scale을 넘기는 방식을 scipy 버전에 따라 오류가 발생하므로 표준화 방식으로 동일한 검정을 수행
        mu = resid.mean()          # 잔차 평균
        sigma = resid.std(ddof=1)  # 잔차 표준편차(표본분산)
        z = (resid - mu) / sigma   # 잔차 표준화
        s, p = kstest(z, 'norm')   # 표준정규분포와 비교한 K-S 검정 통계량 및 p값

    s = float(s)                   # 검정 통계량
    p = float(p)                   # p - value
    normality = bool(p > alpha)      # 정규성 가정 충족 여부 (True, False)

    ### --- 2) 검정 통계량 결과표 ---
    test_df = DataFrame([{
        "statistic": round(s,4),
        "p-value":round(p,4),
        "normality":normality,
        "result":"귀무가설 채택 → 정규성 만족" if normality else "대립가설 채택 → 정규성 위배"
    }], index=[method])

    display(test_df)

    ### --- 3) Q-Q 플롯 ---
    if plot:
        # 팔레트가 지정되면 첫 번째 색을 Q-Q 기준선 색상으로 사용
        line_color = sb.color_palette(palett)[0] if palett else 'red'

        # 잔차를 z-score 표준화한 뒤 Q-Q 플롯용 분위수 계산
        (theoretical, sample),_ = probplot(zscore(resid))

        # Q-Q 플롯용 데이터 프레임 생성
        qq_df = DataFrame({'qq_x': theoretical, 'qq_y':sample})

        # Q-Q 플롯 그리기
        my_plot.lmplot(data=qq_df, x='qq_x', y='qq_y',
                       linecolor=line_color, linestyle="--",
                       xlabel='이론 분위수(Theoretical Quantiles)',
                       ylabel='표본 분위수(Sample Quantiles)',
                       width=width, height=height)

    ### --- 4) √MSE 구간 규칙(68-95-99.7) 판정 ---
    # 잔차 표준편차 추정치 √MSE 를 기준으로 ±1·±2·±3√MSE 구간의 실제 포함 비율을 구하고,
    # '기대 비율 ±2SE'(SE=√(p(1-p)/n)) 허용 범위 안에 드는지 확인한다.
    sqrt_mse = float(np.sqrt(fit.mse_resid))
    expected = [0.68, 0.95, 0.997]      # ±1·±2·±3√MSE 구간의 정규분포 기대 비율

    ratios = []          # 구간별 실제 포함 비율(%) - 잔차도 주석용
    mse_rows = []        # 구간별 판정 상세 (판정표용)
    mse_pass = []        # 구간별 규칙 충족 여부
    for k, exp in zip((1, 2, 3), expected):
        # 해당 구간에 포함된 잔차의 실제 비율
        actual = float(((resid > -k * sqrt_mse) & (resid < k * sqrt_mse)).sum() / n)
        ratios.append(actual * 100)

        # 기대 비율의 표준오차(±2SE)로 허용 범위 산출 후 [0, 1]로 클리핑
        se = np.sqrt(exp * (1 - exp) / n)
        lo = max(0.0, exp - 2 * se)
        hi = min(1.0, exp + 2 * se)
        ok = bool(lo <= actual <= hi)

        mse_pass.append(ok)
        mse_rows.append({
            "구간": f"±{k}√MSE",
            "기대(%)": round(exp * 100, 1),
            "허용범위(%)": f"{lo * 100:.0f}~{hi * 100:.0f}",
            "실제(%)": round(actual * 100, 2),
            "판정": "충족" if ok else "위배",
        })

    
    mse_df = DataFrame(mse_rows).set_index("구간")
    display(mse_df)

    mse_rule = bool(all(mse_pass))        # 세 구간 모두 충족해야 규칙상 정규성 부합
    print(f'√MSE = {sqrt_mse:.2f} / 구간 규칙 판정: {'정규성 부합' if mse_rule else '정규성 위배'}')

    ### --- 5) √MSE 잔차도 (적합값 대비 잔차 ±√MSE 구간) ---
    if plot:
        # 팔레트가 지정되면 3색을 뽑아 ±√MSE 구간 색상으로, 가운데 색을 산점도 색상으로 사용
        band_colors = (sb.color_palette(palett, n_colors=3) if palett
                    else ["#0B3C5D", "#328CC1", "#D9EAF7"])
        point_color = band_colors[1] if palett else "#328CC1"

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

#====================================
# 등분산성 함수 정의 및 파이프라인 구성
#====================================
def test_equalvar(fit, alpha=0.05):
    """
    잔차의 등분산성을 검정한다.

    Args:
        fit: 'fit_model' 함수로 적합된 회귀분석 결과 객체
        alpha (float): 유의수준(기본값: 0.05)
    """
    # --- 1) Breusch-Pagan 검정 (LM/F 통계량) ---
    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(fit.resid, fit.model.exog)
    f_p = float(f_p)
    homoscedasticity = bool(f_p > alpha)

    # --- 2) 두 임계값(alpha, strict_alpha)을 비교한 결과 해석 문자열 ---
    if f_p <= alpha:
        conclusion = f'대립가설 채택 → 등분산 아님'
    else:
        conclusion = f'귀무가설 채택 → 등분산성 만족'

    # --- 3) 단일 행 결과표 구성 및 반환 ---
    result_df = DataFrame([{
        'LM statistic' : round(float(lm_stat), 4),
        'LM p-value':round(float(lm_p),4),
        'F statistic' : round(float(f_stat),4),
        'F p-value' : round(f_p,4),
        'homoscedasticity' : homoscedasticity,
        'result' : conclusion
    }], index=['Breusch-Pagan'])

    display(result_df)  # 결과표 출력


#=======================================
# 독립성 가정 검정 함수 및 파이프라인 구성
#=======================================
def test_independent(fit):
    """
    잔차의 독립성을 검정한다.
    Durbin-Watson 검정은 본래 시계열 데이터 전용이므로, 시간 순서가 없는 데이터에서 독립성이 위배되더라도 무시되는 경우가 많다.
    시각화가 필요하지 않은 검정이므로 plot 파라미터를 제공하지 않는다.

    Args:
        fit : 'fit_model' 함수로 적합된 회귀분석 결과 객체
    """
    # --- 1) Durbin-Watson 통계량 계산 ---
    dw = float(durbin_watson(fit.resid))

    # --- 2) DW값에 따른 독립성 판정 및 해석 ---
    if 1.5 <= dw <=2.5:
        independence = True
        conclusion = '독립성 만족'
    elif dw < 1.5:
        independence = False
        conclusion = '독립성 위반 (양(+)의 자기상관)'        
    else:
        independence = False
        conclusion = '독립성 위반 (음(-)의 자기상관)' 

    # --- 3) 단일 행 결과표 구성 및 출력 ---
    result_df = DataFrame([{
            'statistic':round(dw,4),
            'independence':independence,
            'result':conclusion        
    }], index=['Durbin-Watso'])

    display(result_df) # 결과표 출력

#====================================
# 모형 적합도 - 함수 정의
#====================================
def report_fitness(fit, log_y=False, log_x=None, log1p_y=False, log1p_x=None):
    """
    적합된 회귀모델의 모형 적합도(model fit)를 학술보고 형식의 문장으로 생성해 반환한다.

    summary() 결과표의 문자열을 파싱하지 않고, 'fit' 객체가 이미 갖고 있는 속성에서 지표를 직접 읽어와 문장을 구성한다.
    표에 보이는 값은 반올림된 표시값이지만 'fit'은 완전한 정밀도의 실수값을 갖고 있으므로 보고 형식에 맞춰 round()로 자리수만 맞춘다.

    변환이 적용된 변수는 문장에 log(...)/log1p(...)로 표기하여 실제 적합한 모형을 그래도 드러낸다. 
    (로그 척도에서는 R²도 변환된 종속변수의 분산 설명 비율을 뜻한다.)
    
    Args:
        fit: 'fit_model' 함수로 적합된 회귀분석 결과 객체
        log_y (bool): 종속변수에 로그변환(log)을 적용했는지 여부(기본값: False)
        log_x (list|None): log 변환을 적용한 독립변수 이름 리스트(기본값: None)
        log1p_y (bool) : 종속변수에 log1p(=ln(1+y)) 변환을 적용했는지 여부(기본값: False)
        log1p_x (list|None): log1p 변환을 적용한 독립변수 이름 리스트(기본값: None)

    Returns:
        str: 모형적합도 보고 문장. 'Ipython.display.Markdown'으로 감싸 출력하면 좋다.
    """

    # --- 1) 변수 라벨 구성 ---
    # log_x, log1p_x는 정확한 독립변수 이름 리스트로 전달된다고 가정한다.
    log_x = log_x or []
    log1p_x = log1p_x or []

    # 상수항(const)을 제외한 독립변수 이름 (위치가 아니라 이름ㅇ르 걸러낸다)
    xnames = []
    for name in fit.model.exog_names:
        if name != 'const':
            xnames.append(name)

    # 변환이 적용된 변수는 문장에 log(...)/log1p(...)로 표기한다.
    yname = fit.model.endog_names
    if log1p_y: ylabel = f'log1p({yname})'
    elif log_y: ylabel = f'log({yname})'
    else      : ylabel = yname

    xlabels = [] # 독립변수별 표기 라벨
    for x in xnames:
        if x in log1p_x: xlabels.append(f'log1p({x})')
        elif x in log_x: xlabels.append(f'log({x})')
        else:            xlabels.append(x)

    xlabels = ", ".join(xlabels)

    # --- 2) 유의확률 구간 표기 변환 ---
    if fit.f_pvalue < 0.001:
        alpha = "< 0.001"
    elif fit.f_pvalue < 0.01:
        alpha = "< 0.01"
    elif fit.f_pvalue <0.05:
        alpha = "< 0.05"
    else:
        alpha = "≥ 0.05"

    # --- 3) 문장 템플릿 구성 ---
    # summary() 표를 파싱하지 않고 fit 속성에서 값을 직접 가져오며, 표시값과 동일하게 보이도록 round() 자리수만 맞춘다.
    # Durbin-Watson은 가중잔차(wresid) 기반 계산값
    template = (
        "**Note. n = {n}. "
        "F({df_model}, {df_resid}) = {f_value}, "
        "p {alpha}, "
        "R² = {r_squared}, "
        "Adj.R² = {adj_r_squared}, "
        "Durbin_Watson = {durbin_watson}**\n\n"
        "{Y}를 종속변수로, {X}(을)를 독립변수로 한 {type}회귀분석 결과, 모형은 통계적으로 {result}.\n\n"
        "> F({df_model}, {df_resid}) = {f_value} , p {alpha}, R² = {r_squared}.\n\n"
        "즉, {X}는 {Y}의 약 {r_squared_percent}%를 설명하는 것으로 나타났다.")

    # --- 4) 회귀유형, 유의수준 판별 ---
    # 독립변수 개수로 회귀분석 유형 판별
    if len(xnames) == 1: reg_type = "단순선형"
    else:                reg_type = "다중선형"

    # 유의수준(0.05) 기준 모형의 통계적 유의성 판정
    if fit.f_pvalue < 0.05: result = "유의하였다"
    else:                   result = "유의하지 않았다"

    # --- 5) 문장 템플릿 값 치환 ---
    report = template.format(
        n=int(fit.nobs),                                   
        df_model=int(fit.df_model),                        
        df_resid=int(fit.df_resid),                        
        f_value=round(fit.fvalue, 2),                      
        f_statistic=round(fit.fvalue, 2),
        alpha=alpha,                              
        r_squared=round(fit.rsquared, 3),                  
        adj_r_squared=round(fit.rsquared_adj, 3),          
        durbin_watson=round(durbin_watson(fit.wresid), 3), 
        Y=ylabel,                           
        X=xlabels,                               
        type=reg_type,
        result=result,                                                    
        r_squared_percent=round(fit.rsquared * 100, 2))

    # --- 6) 결과 리턴 ---
    return report


#====================================
# 독립변수 보고 표 생성 - 함수 정의
#====================================
def report_variables(fit, data, hc3=False):
    """
    적합된 회귀모델의 독립변수별 회귀계수 보고표를 데이터프레임으로 생성해 반환한다.

    계수관련 수치는 summary() 표의 반올림된 표시값을 파싱하는 대신 'fit' 객체에서 완전한 정밀도의 실수값으로 직접 가져온다.
    표준화된 회귀계수와 공차, VIF 계산에는 원본 데이터의 표준편차가 필요하므로 'data'를 함께 받는다.

    Args:
        fit: 'fit_model' 함수로 적합된 회귀분석 결과 객체
        data: 회귀분석에 사용한 원본 데이터프레임. 독립변수와 종속변수를 모두 포함해야 한다.

    Returns:
        DataFrame : 독립변수별 보고표. 종속변수, 독립변수, B, 표준오차, β, t, 유의확률, 공차, VIF 컬럼을 갖는다.
    """
    # --- 1) 대상 변수 확인 및 VIF 계산 ---
    yname = fit.model.endog_names    # 종속변수 이름
    # 상수항(const)을 포함한 전체 변수 이름 순서(위치 인덱스 계산에 사용)
    exog_names = list(fit.model.exog_names)
    # 상수항(const)을 제외한 독립변수 이름
    xnames = [name for name in exog_names if name != 'const']

    # 독립변수 전체를 대상으로 VIF를 한 번에 계산(상수항 제외한 결과가 반환된다)
    vif = my_stats.compute_vif(data, columns=xnames)

    # 일반 OLS 통계량일 위치 인덱스로 접근하기 위해 배열로 변환
    params = np.asarray(fit.params)
    bse = np.asarray(fit.bse)
    tvalues = np.asarray(fit.tvalues)
    pvalues = np.asarray(fit.pvalues)

    # -- (신규) cov_type 지정 시 로버스트 표준오차, t, 유의확율을 별도로 계산 --
    # 일반값을 덮어쓰지 않고 비교용으로 따로 보관한다. 
    # t는 정의상 계수/표준오차이므로 표준오차가 로버스트로 바뀌면 t도 함께 바뀐다 (t = B / 로버스트 SE)
    # 유의확률도 이 로버스트 t에서 나온다. 따라서 로버스트 SE, t, 유의확률을 한세트로 가져온다.
    # 로버스트 결과 객체 역시 이름 없는 배열로 반환되므로 동일하게 위치 인덱스로 접근한다.

    if hc3:
        robust = fit.get_robustcov_results(cov_type='HC3')
        rob_bse = np.asarray(robust.bse)
        rob_tvalues = np.asarray(robust.tvalues)
        rob_pvalues = np.asarray(robust.pvalues)



    # --- 2) 독립변수별 계수 및 통계량 정리 ---
    variables = []  # 독립변수를 저장할 빈 리스트
    for x in xnames:
        # 미리 계산해 둔 VIF 표에서 해당 독립 변수의 값을 조회
        vif_value = vif.loc[x, "VIF"]
        i = exog_names.index(x)  # 상수항을 포함한 전체 순서에서의 위치
        b = float(params[i])     # 비표준화 회귀계수 (B)
        # 표준화 회귀계수(β) = B X (독립변수 표준편차 / 종속변수 표준편차)
        beta = b * (data[x].std(ddof=1) / data[yname].std(ddof=1))

        if hc3:
            # 로버스트 비교 형식 : B(양쪽 공유) + 일반(SE, t, 유의확률) + 로버스트(SE, t, 유의확률)를 대칭으로 배치한다.
            # 각 방식의 SE, t, 유의확률이 한 세트로 대응된다.
            row = {
            "종속변수" : yname,                                                                 
            "독립변수" : x,                                                                     
            "B": b,                                                               
            "표준오차": bse[i],
            "표준오차(HC3)" : rob_bse[i],                                                            
            "β": beta, 
            "t": tvalues[i],
            "t(HC3)": rob_tvalues[i],                                                             
            "유의확률": pvalues[i],
            "유의확률(HC3)":rob_pvalues[i],                                                 
            "공차": 1/ vif_value,                                                     
            "VIF": vif_value
            }
        else:
            row = {
            "종속변수" : yname,                                                                 # 종속변수 이름
            "독립변수" : x,                                                                     # 독립변수 이름
            "B": b,                                                                # 비표준화 회귀계수(B)
            "표준오차": bse[i],                                                             # 계수 표준오차
            "β": beta, 
            "t": tvalues[i],                                                               # T-통계량
            "유의확률": pvalues[i],                                                         # 계수 유의확률
            "공차": 1/ vif_value,                                                               # 공차
            "VIF": vif_value 
                }
        
        variables.append(row)


    # --- 2) 독립변수별 계수 및 통계량 정리 ---
    vdf = DataFrame(variables)

    # 베타의 절대값으로 내림차순 정렬 후 리턴(영향력이 큰 변수가 위로 오도록)
    vdf = vdf.sort_values("β", key=abs, ascending=False).reset_index(drop=True)
    return vdf



#====================================
# 회귀계수 보고 문장 생성 함수 
#==================================== 
def report_variables_text(fit, log_y=False, log_x=None, log1p_y=False, log1p_x=None, hc3=False):
    """독립변수별 회귀계수 해석 문장을 markdown 불릿 리스트로 생성해 반환한다.

    Args:
        fit: `fit_model` 함수로 적합된 회귀분석 결과 객체.
        log_y (bool): 종속변수에 로그변환(log)을 적용했는지 여부 (기본값: False).
        log_x (list | None): log 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        log1p_y (bool): 종속변수에 log1p(=ln(1+y)) 변환을 적용했는지 여부 (기본값: False).
        log1p_x (list | None): log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        hc3 (bool) : True면 HC3 로버스트 표준오차를 사용한다. (기본값 : False)

    Returns:
        str: 독립변수별 해석 문장 불릿 리스트. `IPython.display.Markdown`으로 감싸 출력하면 좋다.
    """
    # --- 1) 해석 대상 결정 ---
    # log_x, log1p_x는 정확한 독립변수 이름 리스트로 전달된다고 가정한다.
    log_x = log_x or []
    log1p_x = log1p_x or []

    # 종속변수의 변환 종류 판별
    if log1p_y:
        y_kind = "log1p"
    elif log_y:
        y_kind = "log"
    else:
        y_kind = "none"

    y_pct = y_kind in ("log", "log1p")  # 종속변수가 비율(%) 해석 대상인가

    yname = fit.model.endog_names        # 종속변수 이름
    # 상수항(const)을 제외한 독립변수 이름 (위치가 아니라 이름으로 걸러낸다)
    xnames = []
    for name in fit.model.exog_names:
        if name != "const":
            xnames.append(name)

    df_resid = int(fit.df_resid)         # t 분포 자유도(잔차 자유도)

    # 종속변수 쪽 % 해석의 대상: log1p는 (1+y), 그 외는 y
    if y_kind == "log1p":
        y_target = f"**(1+{yname})**"
    else:
        y_target = yname

    # -- (신규) hc3=True인 경우 로버스트 표준오차 기반 t/유의확률로 교체 --
    # 회귀계수(B)는 그대로이고, 표준오차만 이분산에 강건한 HC3로 바뀐다.
    # t = B / 로버스트 SE 이므로 t와 유의확률도 한세트로 함께 바뀐다.
    # 로버스트 결과 객체는 이름없는 배열을 반환하으모 위치 인덱스로 접근한다.
    if hc3:
        robust = fit.get_robustcov_results(cov_type='HC3')
        rob_tvalues = np.asarray(robust.tvalues)
        rob_pvalues = np.asarray(robust.pvalues)

    # --- 2) 문장 템플릿 구성 (독립변수마다 반복 적용) ---
    line_template = (
        "- **{x}**의 회귀계수는 **B = {B}**으로 나타났으며, "
        "이는 **{y}**에 {sig} 요인임을 의미한다. "
        "(**t({df}) = {t}**, **{p}**) "
        "즉, {effect} 것으로 해석된다."
    )
    effect_template = "{x_change} {y_target}는 평균적으로 {approx}**{mag}{unit} {direction}**하는"

    # --- 3) 독립변수별 해석 문장 생성 ---
    lines = []    # 독립변수별 문장(불릿)을 저장할 빈 리스트
    for x in xnames:
        # 이 독립변수의 변환 여부 (정확한 이름 리스트라고 가정)
        x_is_log1p = x in log1p_x
        x_is_log = x in log_x
        x_pct = x_is_log or x_is_log1p    # 독립변수가 비율(%) 증가 기준인가
        B = fit.params[x]                 # 비표준화 회귀계수(B)

        if hc3:
            # 로버스트(HC3) 표준오차에서 나온 t, 유의확률로 유의성을 판정한다.
            i= fit.model.exog_names.index(x)  # 상수항을 포함한 전체 순서에서의 위히
            t= float(rob_tvalues[i])  # 로버스트 t (=B / 로버스트 SE)
            p = float(rob_pvalues[i]) # 로버스트 유의확률

        else:
            t = fit.tvalues[x]                # t-통계량
            p = fit.pvalues[x]                # 계수 유의확률

        # 유의성 판정 (유의수준 0.05 기준)
        if p < 0.05:
            sig_word = "유의한"
        else:
            sig_word = "유의하지 않은"

        # p값 APA 표기 (앞자리 0 생략)
        if p < 0.001:
            p_text = "p < .001"
        else:
            p_text = f"p = {p:.3f}".replace("0.", ".")

        # 계수 부호로 증가/감소 방향 결정
        if B > 0:
            direction = "증가"
        else:
            direction = "감소"
        
        # 독립변수 변화 표현: log1p는 (1+x)가 1% 증가, log는 x가 1% 증가, 원변은 x가 1 증가
        if x_is_log1p: x_change = f"**(1+{x})가 1% 증가**할 때"
        elif x_is_log: x_change = f"**{x}가 1% 증가**할 때"
        else:  x_change = f"**{x}가 1 증가**할 때"

        # 효과 크기: x·y가 각각 비율(%) 기준인지에 따라 값·단위가 정해진다
        if not x_pct and not y_pct:
            # 원본 → 원본
            mag, unit, approx = f"{abs(B):.2f}", "", ""
        elif x_pct and not y_pct:
            # 독립변수만 로그 → 1% 증가당 절대 변화 ≈ B×ln(1.01)
            mag, unit, approx = f"{abs(B * np.log(1.01)):.3f}", "", "약 "
        elif not x_pct and y_pct:
            # 종속변수만 로그 → (e^B - 1)×100 %
            mag, unit, approx = f"{abs((np.exp(B) - 1) * 100):.2f}", "%", "약 "
        else:
            # 둘 다 로그 → 탄력성 B %
            mag, unit, approx = f"{abs(B):.2f}", "%", "약 "

        effect = effect_template.format(
            x_change=x_change,
            y_target=y_target,
            approx=approx,
            mag=mag,
            unit=unit,
            direction=direction)

        # 하나의 독립변수 = 하나의 불릿 문장
        lines.append(line_template.format(
                x=x,
                B=round(B, 2),
                y=yname,
                sig=sig_word,
                df=df_resid,
                t=round(t, 2),
                p=p_text,
                effect=effect,
            ))     
    
    # --- 4) 로버스트 표준오차 및 log1p 사용 시 해석 주의 각주 첨부 ---
    report = "\n".join(lines)

    if hc3:
        report += (
            "\n\n> ※ 위 **t**와 **유의확률**은 등분산 가정이 충족되지 않은 경우를 대비해 "
            "**HC3 로버스트 표준오차**로 계산한 값이다. 회귀계수(B)와 효과 크기 해석은 "
            "일반 OLS와 동일하며, 표준오차만 이분산에 강건하게 보정되어 유의성 판정이 달라질 수 있다."
        )

    uses_log1p = (y_kind == "log1p") or bool(log1p_x)
    if uses_log1p:
        report += (
            "\n\n> ※ **log1p**(=ln(1+.))의 % 해석은 변수 자체가 아니라 **(1+변수)** 기준이며, "
            "값이 클 때만 위 근사가 성립한다(0·작은 값 구간에서는 원본처럼 동작해 부정확). "
            "이 구간에서는 부호·유의성 중심으로 해석하거나 예측값을 expm1로 원 척도에서 비교한다."
        )

    return report


#====================================
# 선형회귀 일괄 처리 함수 정의 
#==================================== 
def auto_ols(data, y, summary=False, report=True,
             log_y=False, log_x=None, log1p_y=False, log1p_x=None,
             test=True, plot=False, width=1280, height=640,
             backward=False, alpha=0.05):
    """회귀모델 적합부터 보고서 출력·가정 검정까지 한 번에 수행한다.

    Args:
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임.
        y: 종속변수로 사용할 컬럼명.
        summary (bool): 적합 모델의 statsmodels 요약 통계량 출력 여부 (기본값: False).
        report (bool): 모형 적합도 보고서 출력 여부 (기본값: True).
        log_y (bool): 종속변수에 log 변환을 적용했는지 여부 (기본값: False).
        log_x (list | None): log 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        log1p_y (bool): 종속변수에 log1p 변환을 적용했는지 여부 (기본값: False).
        log1p_x (list | None): log1p 변환을 적용한 독립변수 이름 리스트 (기본값: None).
        test (bool): 회귀모형 가정 검정 수행 여부 (기본값: True).
        plot (bool): 가정 검정 시 그래프를 함께 그릴지 여부 (기본값: False).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
        backward (bool): True이면 후진소거법으로 유의하지 않은 독립변수를 제거한 뒤
                         최종 모델을 적합한다 (기본값: False).
        alpha (float): 후진소거법의 변수 제거 기준 유의수준 (기본값: 0.05).

    Returns:
        적합이 완료된 회귀분석 결과 객체.
    """

    # # --- 1) 회귀모델 적합 ---
    # fit = fit_model(data, y, summary=summary)

    # # 빈 줄 출력 (출력 결과의 여백을 위함)
    # print()

    # # --- 2) 회귀모형 적합도 보고서 출력 ---
    # # 등분산성 가정 확인
    # lm_stat, lm_p, f_stat, f_p = het_breuschpagan(fit.resid, fit.model.exog)
    # # 등분산 충족시 True, 위배시 False (유의수준 0.05 기준)
    # homoscedasticity = bool(float(f_p) >= 0.05)

    # 빈 줄 출력 (출력 결과의 여백을 위함)
    print()

    while True:
        # --- 1) 회귀모델 적합 (summary는 최종 모델에서만 출력한다) ---
        fit = fit_model(data, y)

        # --- 2) 등분산성 가정 확인 ---
        lm_stat, lm_p, f_stat, f_p = het_breuschpagan(fit.resid, fit.model.exog)
        # 등분산 충족시 True, 위배시 False (유의수준 0.05 기준)
        homoscedasticity = bool(float(f_p) >= 0.05)

        if not backward:
            break  # 후진소거법이 아니면 반복문 종료

        report_vars = report_variables(fit, data, hc3=not homoscedasticity)
        # 등분산이면 일반 OLS의 유의확률, 위배되면 HC3 유의확률을 제거 기준으로 삼는다
        pvalues = report_vars["유의확률"] if homoscedasticity else report_vars["유의확률(HC3)"]

        # 독립변수가 하나뿐이거나 모두 유의하면 종료
        if len(pvalues) <= 1 or pvalues.max() <= alpha:
            break

        # 유의확률이 가장 큰(=가장 유의하지 않은) 독립변수를 하나만 제거한다.
        # 여러 개를 한꺼번에 지우면, 변수 간 상관 때문에 원래는 유의해졌을 변수까지 사라진다.
        worst = report_vars.loc[pvalues.idxmax(), "독립변수"]
        print(f"유의하지 않은 독립변수 제거 → {worst} (p = {pvalues.max():.4f})")
        data = data.drop(columns=[worst])

    # 최종 모델의 요약 통계량 출력
    if summary:
        print(fit.summary())

    if report:
        display(Markdown("#### ▶ 모형 적합도"))
        # 회귀계수 보고 표(hc3는 등분산 충족 아닐 시 True로 설정)
        display(report_variables(fit, data, hc3=not homoscedasticity))
        display(Markdown(report_fitness(fit, log_y=log_y, log_x=log_x,
                                        log1p_y=log1p_y, log1p_x=log1p_x)))

    # --- 3) 회귀모형 가정 검정 ---
    # 보고서와 가정 검정이 모두 출력되는 경우, 구분을 위해 수평선 추가
    if report and test:
        display(Markdown("---"))

    # 회귀모형 가정 검정 (선형성 → 정규성 → 등분산성 → 독립성)
    if test:
        display(Markdown("#### ▶ 회귀모형 가정 검정"))
        display(Markdown("##### 1) 선형성 검정"))
        test_linear(fit, plot=plot, width=width, height=height)
        display(Markdown("##### 2) 정규성 검정"))
        test_normal(fit, plot=plot, width=width, height=height)
        display(Markdown("##### 3) 등분산성 검정"))
        test_equalvar(fit)
        display(Markdown("##### 4) 독립성 검정"))
        test_independent(fit)

    # --- 4) 최종 적합 모델 객체 반환 ---
    return fit


#====================================
# 독립변수의 영향력 순위 시각화 함수 
#==================================== 
def plot_beta(fit, data, palette=None, title=None, xlabel=None, ylabel=None, width=1280, height=None, save_path=None):
    """
    표준화 회귀계수(β)를 가로 막대그래프로 시각화해 독립변수의 영향력 순위를 보여준다.

    β의 절대값 순위는 종속변수에 미치는 영향력의 순위를 의미한다(영향력의 절대적 크기는 아니다).
    막대는 'report_variables'가 정렬해 둔 |β| 내림차순 그대로 위에서 아래로 배치되며,
    계수의 부호에 따라 색을 달리하고 막대 끝에 β 값을 표기한다.

    Args:
        fit: 'fit_model' 함수로 적합된 회귀분석 결과 객체
        data: 독립변수와 종속변수를 모두 포함하는 데이터프레임
        palette (dict): 부호별 막대색상. None이면 {+:파랑, -:빨강} (기본값:None)
        title (str): 그래프 제목 (기본값:None)
        xlabel (str): x축 레이블 (기본값:None → '표준화 계수(β)')
        ylabel (str): y축 레이블 (기본값:None → '독립변수')
        width (int): 캔버스 가로 픽셀 (기본값: 1280)
        height (int): 캔버스 세로 픽셀. None이면 독립변수 수 x 80으로 자동 계산 (기본값: None)
        save_path (str): 이미지 저장 경로 (기본값:None)
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