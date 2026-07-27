# =====================================================================
# 로그 변환 — 치우친 분포를 대칭에 가깝게 편다
# =====================================================================
def log_transform(df, log_columns=None, log1p_columns=None, reflect_columns=None, verbose=True):
    """
    로그변환 함수

    Args:
        df (DataFrame): 변환을 적용할 데이터프레임
        log_columns (list, optional): 순수 로그 변환할(우측 꼬리, 0 없음) 컬럼명 리스트 (기본값: None)
        log1p_columns (list, optional): log(1+x) 변환할(우측 꼬리) 컬럼명 리스트 (기본값: None)
        reflect_columns (list, optional): 반사 후 로그 변환할(좌측 꼬리) 컬럼명 리스트 (기본값: None)
        verbose (bool): 컬럼별 변환식·역변환식과 왜도 변화를 출력할지 여부 (기본값: True)

    Returns:
        DataFrame: 변환이 적용된 데이터프레임 (원본은 변경되지 않는다)

    Raises:
        ValueError: `log_columns` 에 0 이하의 값이 있어 순수 로그를 취할 수 없는 경우
    """
    # --- 1) 작업 준비 ---
    result = df.copy()    # 원본을 보존하기 위해 복사본으로 작업
    report = []           # verbose 출력을 위해 컬럼별 변환 내역을 기록

    # --- 2) 우측 꼬리 컬럼 변환 (1): log(x) ---
    # 0 이하의 값이 하나라도 있으면 -inf 나 NaN 이 되므로 미리 막는다
    if log_columns:
        for c in log_columns:
            if df[c].min() <= 0:
                raise ValueError(f"'{c}' 컬럼에 0 이하의 값이 있어 log(x)를 취할 수 없습니다. "
                                 f'(최솟값 {df[c].min():g}) log1p_columns 로 넘기세요.')

            result[c] = np.log(df[c])
            report.append([c, '우측 꼬리', 'log(x)', 'exp(y)'])

    # --- 3) 우측 꼬리 컬럼 변환 (2): log(1+x) ---
    # 값이 0인 경우 log(0) = -inf 가 되므로 log(1+x) 를 사용한다
    if log1p_columns:
        for c in log1p_columns:
            result[c] = np.log1p(df[c])
            report.append([c, '우측 꼬리', 'log(1+x)', 'exp(y)-1'])

    # --- 4) 좌측 꼬리 컬럼 변환: 반사 후 log(1+x) ---
    # 최댓값에서 빼면 좌우가 뒤집혀 좌측 꼬리가 우측 꼬리가 되므로, 그 뒤 동일하게 로그를 취한다
    # 값의 대소 관계가 뒤집히므로 회귀계수의 부호도 반대로 해석해야 한다
    if reflect_columns:
        for c in reflect_columns:
            # 역변환하려면 이 최댓값이 반드시 필요하므로 verbose 출력에 함께 남긴다
            max_value = df[c].max()
            result[c] = np.log1p(max_value - df[c])
            report.append([c, '좌측 꼬리',
                           f'log(1+{max_value:g}-x)',
                           f'{max_value:g}-(exp(y)-1)'])

    # --- 5) 변환 내역 출력 (변환식·역변환식과 왜도 변화) ---
    if verbose:
        print(f'{"컬럼":10s}{"꼬리방향":10s}{"변환식":22s}{"역변환식":24s}{"왜도":>16s}')
        print('-' * 88)

        for c, side, func, inverse in report:
            before = skew(df[c].dropna())
            after = skew(result[c].dropna())
            change = f'{before:.2f} -> {after:.2f}'
            print(f'{c:10s}{side:10s}{func:22s}{inverse:24s}{change:>14s}')

    # --- 6) 변환이 적용된 데이터프레임 반환 ---
    return result



# =====================================================================
# 로그 역변환
# =====================================================================
def inverse_log_transform(df, log_columns=None, log1p_columns=None,
                          reflect_columns=None, verbose=True):
    """
    log_transform() 으로 변환된 컬럼을 원래 값(단위)으로 되돌리는 함수

    Args:
        df (DataFrame): 역변환을 적용할 데이터프레임
        log_columns (list, optional): 순수 로그 변환했던(우측 꼬리) 컬럼명 리스트 (기본값: None)
        log1p_columns (list, optional): log(1+x) 변환했던(우측 꼬리) 컬럼명 리스트 (기본값: None)
        reflect_columns (dict, optional): 반사 변환했던(좌측 꼬리) 컬럼의
            {컬럼명: 변환 당시의 최댓값} (예: {'B': 396.9}) (기본값: None)
        verbose (bool): 컬럼별 역변환식과 값의 범위 변화를 출력할지 여부 (기본값: True)

    Returns:
        DataFrame: 역변환이 적용된 데이터프레임 (원본은 변경되지 않는다)
    """
    # --- 1) 작업 준비 ---
    result = df.copy()    # 원본을 보존하기 위해 복사본으로 작업
    report = []           # verbose 출력을 위해 컬럼별 역변환 내역을 기록

    # --- 2) 우측 꼬리 컬럼 역변환 (1): exp(y) ---
    # log(x) 의 역함수인 exp(y) 로 되돌린다
    if log_columns:
        for c in log_columns:
            result[c] = np.exp(df[c])
            report.append([c, '우측 꼬리', 'exp(y)'])

    # --- 3) 우측 꼬리 컬럼 역변환 (2): exp(y)-1 ---
    # log(1+x) 의 역함수인 exp(y)-1 로 되돌린다
    if log1p_columns:
        for c in log1p_columns:
            result[c] = np.expm1(df[c])
            report.append([c, '우측 꼬리', 'exp(y)-1'])

    # --- 4) 좌측 꼬리 컬럼 역변환: 최댓값 - (exp(y)-1) ---
    # 로그를 먼저 풀고(exp(y)-1), 그 결과를 최댓값에서 빼서 뒤집힌 대소 관계를 되돌린다
    if reflect_columns:
        for c, max_value in reflect_columns.items():
            result[c] = max_value - np.expm1(df[c])
            report.append([c, '좌측 꼬리', f'{max_value:g}-(exp(y)-1)'])

    # --- 5) 역변환 내역 출력 (역변환식과 값의 범위 변화) ---
    if verbose:
        print(f'{"컬럼":10s}{"꼬리방향":10s}{"역변환식":24s}{"값의 범위":>28s}')
        print('-' * 76)

        for c, side, inverse in report:
            before = f'{df[c].min():.2f}~{df[c].max():.2f}'
            after = f'{result[c].min():.2f}~{result[c].max():.2f}'
            change = f'{before} -> {after}'
            print(f'{c:10s}{side:10s}{inverse:24s}{change:>26s}')

    # --- 6) 역변환이 적용된 데이터프레임 반환 ---
    return result