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
        verbose (bool): 단계별 전처리 내역 출력 여부 (기본값: False).

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
