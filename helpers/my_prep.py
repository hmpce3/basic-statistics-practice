import os
import joblib
import numpy as np
from pandas import pivot_table, get_dummies
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from . import my_stats

# scaling() 에서 사용할 스케일러 이름과 클래스의 매핑
SCALERS = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
    'maxabs': MaxAbsScaler,
}

# =====================================================================
# 형태 변환
# =====================================================================
def long2wide(df, hue, values, dropna=True):
    """
    long format 데이터프레임을 group 단위로 컬럼을 펼쳐 wide format으로 변환하는 함수

    Args:
        - df: 변환할 데이터프레임
        - hue: 펼칠 기준이 되는 그룹 열 이름 (각 값이 새 열이 됨)
        - values: 펼칠 값이 담긴 열 이름
        - dropna: 결측치 행을 결과에서 제외할지 여부 (기본값 True)

    Returns
        - wide format으로 변환된 데이터프레임
    """
    wide = pivot_table(data=df,
                       index=df.groupby(hue, observed=True).cumcount(),
                       columns=hue, values=values, dropna=dropna, observed=True)
    wide.columns.name = None
    wide.index.name = None
    return wide


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
    """log_transform() 으로 변환된 컬럼을 원래 값(단위)으로 되돌리는 함수

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


# =====================================================================
# 라벨링 — 범주형 문자열을 정수로 바꾼다
# =====================================================================
def labeling(df, columns, save_path=None, verbose=True):
    """
    지정한 범주형 컬럼들의 값을 0부터 시작하는 정수로 변환하는 함수

    Args:
        df (DataFrame): 라벨링을 적용할 데이터프레임
        columns (list): 라벨링할 컬럼명 리스트
        save_path (str, optional): 학습된 인코더들을 저장할 pkl 파일 경로
            (예: 'models/encoders.pkl') (기본값: None, 저장하지 않음)
        verbose (bool): 컬럼별로 원래 값이 어떤 정수에 대응되는지 출력할지 여부 (기본값: True)

    Returns:
        DataFrame: 라벨링이 적용된 데이터프레임 (원본은 변경되지 않는다)
    """
    # --- 1) 작업 준비 ---
    result = df.copy()    # 원본을 보존하기 위해 복사본으로 작업
    encoders = {}         # 역변환과 test 데이터 적용을 위해 컬럼별 인코더를 보관

    # --- 2) 컬럼별 라벨 인코딩 (문자열 -> 0부터 시작하는 정수) ---
    for c in columns:
        encoder = LabelEncoder()
        result[c] = encoder.fit_transform(df[c])
        encoders[c] = encoder

    # --- 3) 변환 내역 출력 (원래 값 -> 부여된 정수) ---
    if verbose:
        for c in columns:
            # classes_ 의 순서가 곧 부여된 정수값이므로 짝지어 출력한다
            pairs = []
            for i, v in enumerate(encoders[c].classes_):
                pairs.append(f'{v}={i}')

            print(f'{c} ({len(pairs)}종): {", ".join(pairs)}')

    # --- 4) 학습된 인코더 저장 (선택) ---
    # 컬럼마다 인코더가 따로 있으므로 dict 통째로 하나의 pkl 파일에 저장한다
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            # 경로에 없는 폴더가 있으면 만들어 준다
            os.makedirs(folder, exist_ok=True)

        joblib.dump(encoders, save_path)

        if verbose:
            print(f'\n인코더 저장: {save_path} ({len(encoders)}개 컬럼)')

    # --- 5) 라벨링이 적용된 데이터프레임 반환 ---
    return result


# =====================================================================
# 결측치 처리 — 비어 있는 값을 삭제하거나 다른 값으로 채운다
# =====================================================================
def replace_missing(df, columns=None, method='mean', value=None, verbose=True):
    """
    지정한 컬럼들의 결측치를 삭제하거나 다른 값으로 대체하는 함수

    Args:
        df (DataFrame): 결측치를 처리할 데이터프레임
        columns (list, optional): 결측치를 처리할 컬럼명 리스트.
            None 이면 결측치가 있는 컬럼을 자동으로 선택한다 (기본값: None)
        method (str): 결측치를 어떻게 처리할지 지정한다. 대소문자는 구분하지 않는다 (기본값: 'mean')
            - 'drop':   결측치가 있는 행을 삭제
            - 'value':  사용자가 value 파라미터로 지정한 고정값
            - 'mean':   해당 컬럼의 평균   (수치형)
            - 'median': 해당 컬럼의 중앙값 (수치형)
            - 'max':    해당 컬럼의 최댓값 (수치형)
            - 'min':    해당 컬럼의 최솟값 (수치형)
            - 'mode':   해당 컬럼의 최빈값 (수치형·범주형 모두 가능)
        value (any, optional): method='value' 일 때 결측치를 대체할 고정값 (기본값: None)
        verbose (bool): 컬럼별 결측치 개수와 대체값을 출력할지 여부 (기본값: True)

    Returns:
        DataFrame: 결측치가 처리된 데이터프레임 (원본은 변경되지 않는다)
    """
    # --- 1) 처리 대상 컬럼 결정 ---
    name = method.lower().strip()

    # 처리 대상 컬럼 결정: 지정이 없으면 결측치가 있는 컬럼만 자동 선택
    # 단, 평균·중앙값·최댓값·최솟값은 수치형에만 쓸 수 있으므로 수치형 컬럼으로 한정한다
    if columns is None:
        target = df
        if name in ['mean', 'median', 'max', 'min']:
            target = df.select_dtypes(include='number')

        counts = target.isna().sum()
        columns = list(counts[counts > 0].index)

    # --- 2) 작업 준비 ---
    result = df.copy()    # 원본을 보존하기 위해 복사본으로 작업
    report = []           # verbose 출력을 위해 컬럼별 처리 내역을 기록

    # --- 3) 삭제 방식은 행 단위로 한 번에 처리하고 끝낸다 ---
    # 대상 컬럼 중 하나라도 결측이면 그 행을 통째로 지운다
    if name == 'drop':
        result = df.dropna(subset=columns).reset_index(drop=True)

        if verbose:
            print(f"결측치 처리 방식: 'drop' (대상 컬럼 {len(columns)}개)")
            print(f'행 수: {len(df)}개 -> {len(result)}개 ({len(df) - len(result)}개 삭제)')

        return result

    # --- 4) 컬럼별 대체값 계산 및 결측치 대체 ---
    for c in columns:
        count = df[c].isna().sum()

        # 4-1) method 에 따라 대체값을 정한다
        if name == 'mean':
            fill = df[c].mean()
        elif name == 'median':
            fill = df[c].median()
        elif name == 'max':
            fill = df[c].max()
        elif name == 'min':
            fill = df[c].min()
        elif name == 'mode':
            # 최빈값은 동점이면 여러 개가 나오므로 첫 번째 값을 사용한다
            fill = df[c].mode()[0]
        else:
            fill = value

        # 4-2) 정해진 값으로 그 컬럼의 결측치를 채운다
        result[c] = df[c].fillna(fill)
        report.append([c, count, fill])

    # --- 5) 처리 내역 출력 (컬럼별 결측치 개수와 대체값) ---
    if verbose:
        print(f"결측치 처리 방식: '{name}'")
        print(f'{"컬럼":12s}{"결측치":>14s}{"대체값":>20s}')
        print('-' * 44)

        for c, count, fill in report:
            ratio = count / len(df) * 100
            found = f'{count}개({ratio:.1f}%)'
            print(f'{c:12s}{found:>12s}{str(fill):>18s}')

    # --- 6) 결측치가 처리된 데이터프레임 반환 ---
    return result


# =====================================================================
# 이상치 대체 — 극단값을 다른 값으로 바꾼다
# =====================================================================
def replace_outlier(df, columns=None, method='bound', value=None, verbose=True):
    """
    지정한 컬럼들의 이상치를 다른 값으로 대체하는 함수

    Args:
        df (DataFrame): 이상치를 대체할 데이터프레임
        columns (list, optional): 이상치를 대체할 컬럼명 리스트.
            None 이면 df 의 수치형 컬럼을 자동으로 선택한다 (기본값: None)
        method (str): 이상치를 무엇으로 대체할지 지정한다. 대소문자는 구분하지 않는다 (기본값: 'bound')
            - 'bound':  이상치 경계값. 
            - 'median': 해당 컬럼의 중앙값
            - 'mean':   해당 컬럼의 평균
            - 'value':  사용자가 value 파라미터로 지정한 고정값
        value (number, optional): method='value' 일 때 이상치를 대체할 고정값 (기본값: None)
        verbose (bool): 컬럼별 이상치 경계와 대체된 개수를 출력할지 여부 (기본값: True)

    Returns:
        DataFrame: 이상치가 대체된 데이터프레임 (원본은 변경되지 않는다)
    """
    # --- 1) 처리 대상 컬럼 결정 ---
    # 처리 대상 컬럼 결정: 지정이 없으면 수치형 컬럼만 자동 선택
    if columns is None:
        columns = list(df.select_dtypes(include='number').columns)

    # --- 2) 작업 준비 ---
    result = df.copy()    # 원본을 보존하기 위해 복사본으로 작업
    name = method.lower().strip()
    report = []           # verbose 출력을 위해 컬럼별 대체 내역을 기록

    # --- 3) 컬럼별 이상치 판단 및 대체 ---
    for c in columns:
        # 3-1) IQR 기반 이상치 경계 계산
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # 3-2) 경계를 벗어난 값의 위치
        is_outlier = (df[c] < lower) | (df[c] > upper)
        count = is_outlier.sum()

        # 3-3) method 에 따라 대체값을 정하고 이상치를 바꾼다
        if name == 'bound':
            # 하한/상한 바깥의 값을 각각 하한/상한으로 잘라낸다
            result[c] = df[c].clip(lower, upper)
            replaced = f'{lower:.2f} 또는 {upper:.2f}'
        else:
            if name == 'median':
                # 이상치를 제외한 정상값만으로 대푯값을 구해야 이상치에 오염되지 않는다
                fill = df.loc[~is_outlier, c].median()
            elif name == 'mean':
                fill = df.loc[~is_outlier, c].mean()
            else:
                fill = value

            result.loc[is_outlier, c] = fill
            replaced = f'{fill:.2f}'

        report.append([c, lower, upper, count, replaced])

    # --- 4) 대체 내역 출력 (정상 범위·이상치 개수·대체값) ---
    if verbose:
        print(f"이상치 대체 방식: '{name}' (기준: IQR x 1.5)")
        print(f'{"컬럼":12s}{"정상 범위":>26s}{"이상치":>10s}{"대체값":>22s}')
        print('-' * 68)

        for c, lower, upper, count, replaced in report:
            bound = f'{lower:.2f} ~ {upper:.2f}'
            ratio = count / len(df) * 100
            found = f'{count}개({ratio:.1f}%)'
            print(f'{c:12s}{bound:>24s}{found:>12s}{replaced:>20s}')

    # --- 5) 이상치가 대체된 데이터프레임 반환 ---
    return result


# =====================================================================
# 더미 변환 — 범주형을 0/1 컬럼으로 펼친다
# =====================================================================
def dummies(df, columns, drop_first=True, verbose=True):
    """
    지정한 범주형 컬럼들을 값의 종류마다 0/1 컬럼으로 펼치는 함수

    Args:
        df (DataFrame): 더미 변환을 적용할 데이터프레임
        columns (list): 더미 변환할 컬럼명 리스트
        drop_first (bool): 각 컬럼에서 만들어진 더미 중 첫 번째를 제외할지 여부 (기본값: True)
            k개의 값에서 k개의 더미를 모두 만들면 서로의 합이 항상 1이 되어 완전한
            다중공선성(더미 변수 함정)이 생기므로, 하나를 빼서 기준(reference) 범주로 삼는다
        verbose (bool): 컬럼별로 생성된 더미 컬럼과 생략된 컬럼을 출력할지 여부 (기본값: True)

    Returns:
        DataFrame: 더미 변환이 적용된 데이터프레임 (원본은 변경되지 않는다)
    """
    # --- 1) 처리 대상 컬럼 결정 ---
    # 값의 종류가 2개인 컬럼은 변환 대상에서 제외한다
    targets = []
    skipped = []

    for c in columns:
        if df[c].nunique() == 2:
            skipped.append(c)
        else:
            targets.append(c)

    # --- 2) 더미 변환 수행 ---
    # dtype=int 를 지정해 True/False 가 아닌 0/1 로 만든다
    result = get_dummies(df, columns=targets, drop_first=drop_first, dtype=int)

    # --- 3) 변환 내역 출력 (생성된 더미 컬럼과 생략된 컬럼) ---
    if verbose:
        for c in targets:
            # 원본에 없고 결과에만 있으면서 'c_' 로 시작하는 컬럼이 c 로부터 생성된 더미다
            created = []
            for new in result.columns:
                if new not in df.columns and new.startswith(f'{c}_'):
                    created.append(new)

            # drop_first 로 빠진 기준 범주가 무엇인지 함께 알려준다
            dropped = ''
            if drop_first:
                dropped = f'  (기준: {sorted(df[c].unique())[0]} 제외)'

            print(f'{c} ({df[c].nunique()}종) -> {len(created)}개: {", ".join(created)}{dropped}')

        for c in skipped:
            print(f'{c} (2종) -> 생략: 이진 변수이므로 원래 컬럼을 유지')

        print(f'\n컬럼 수: {df.shape[1]}개 -> {result.shape[1]}개')

    # --- 4) 더미 변환이 적용된 데이터프레임 반환 ---
    return result


# =====================================================================
# 다중공선성 제거 — 서로 겹치는 변수를 걸러낸다
# =====================================================================
def reduce_vif(df, columns=None, threshold=10.0, verbose=True):
    """
    다중 공선성이 사라질 때까지 VIF 가 가장 큰 변수를 하나씩 반복 제거하는 함수

    Args:
        df (DataFrame): 다중 공선성을 제거할 변수들이 포함된 데이터프레임
        columns (list, optional): 다중 공선성을 판단/제거할 컬럼명 리스트.
            None 이면 df 의 수치형 컬럼을 자동으로 선택한다 (기본값: None)
        threshold (float): 다중 공선성 판단 기준이 되는 VIF 임계값 (기본값: 10.0)
        verbose (bool): 제거 과정과 결과를 단계별로 출력할지 여부 (기본값: True)

    Returns:
        DataFrame: 대상 변수들의 VIF 가 모두 threshold 미만이 되도록 일부 변수가
                   제거된 데이터프레임. 대상이 아닌 컬럼은 원래 순서대로 보존된다.
    """
    # --- 1) 처리 대상 컬럼 결정 ---
    # 처리 대상 컬럼 결정: 지정이 없으면 수치형 컬럼만 자동 선택
    if columns is None:
        targets = list(df.select_dtypes(include='number').columns)
    else:
        missing = []
        for c in columns:
            if c not in df.columns:
                missing.append(c)

        if missing:
            raise KeyError(f'df 에 존재하지 않는 컬럼입니다: {missing}')
        targets = list(columns)

    # 컬럼이름의 오름차순으로 정렬
    targets.sort()

    # 대상에서 제외되는 컬럼(종속변수, 명목형 등)은 그대로 보존하기 위해 컬럼이름을 따로 기록
    keep = []
    for c in df.columns:
        if c not in targets:
            keep.append(c)

    # --- 2) 반복 제거 과정 ---
    work = df[targets].copy()    # 원본을 보존하기 위해 대상 변수만 복사본으로 작업
    step = 0                     # 반복 단계 카운터

    while True:
        vif = my_stats.compute_vif(work)
        max_vif = vif['VIF'].max()

        # 종료 조건: 가장 큰 VIF도 기준 미만이거나 남은 변수가 한개라면 종료
        if max_vif < threshold or len(work.columns) <= 1:
            print(f'\n완료! 남은 변수: {list(work.columns)}')
            print(f'최대 VIF = {max_vif:.2f}')
            break

        # 가장 VIF 가 큰 변수를 찾아 제거하고 다시 반복
        worst = vif['VIF'].idxmax()
        step += 1
        if verbose:
            print(f'[{step}단계] {worst} 제거 (VIF = {max_vif:.1f})')
        work = work.drop(columns=[worst])

    # --- 3) 보존 대상 컬럼과 합쳐 원래 컬럼 순서를 유지해 반환 ---
    survived = []
    for c in df.columns:
        if c in keep or c in work.columns:
            survived.append(c)

    return df[survived]


# =====================================================================
# 정규화 — 변수들의 값의 범위를 통일한다
# =====================================================================
def scaling(df, columns=None, method='standard', save_path=None, verbose=True):
    """
    지정한 컬럼들의 값의 범위(스케일)를 통일하는 함수

    Args:
        df (DataFrame): 스케일링을 적용할 데이터프레임
        columns (list, optional): 스케일링할 컬럼명 리스트.
            None 이면 df 의 수치형 컬럼을 자동으로 선택한다 (기본값: None)
        method (str): 사용할 스케일러 이름. 대소문자와 뒤의 'Scaler' 는 무시하므로
            'standard', 'Standard', 'StandardScaler' 를 모두 같게 취급한다 (기본값: 'standard')
            - 'standard': (x - 평균) / 표준편차 -> 평균 0, 표준편차 1
            - 'minmax':   (x - 최소) / (최대 - 최소) -> 0 ~ 1
            - 'robust':   (x - 중앙값) / IQR -> 이상치의 영향을 덜 받음
            - 'maxabs':   x / |최대| -> -1 ~ 1 (0을 보존하므로 희소 데이터에 사용)
        save_path (str, optional): 학습된 스케일러를 저장할 pkl 파일 경로
            (예: 'models/scaler.pkl') (기본값: None, 저장하지 않음)
        verbose (bool): 컬럼별 스케일링 전후의 값의 범위를 출력할지 여부 (기본값: True)

    Returns:
        DataFrame: 스케일링이 적용된 데이터프레임 (원본은 변경되지 않는다)
    """
    # --- 1) 처리 대상 컬럼 결정 ---
    # 처리 대상 컬럼 결정: 지정이 없으면 수치형 컬럼만 자동 선택
    if columns is None:
        columns = list(df.select_dtypes(include='number').columns)

    # --- 2) 작업 준비 및 스케일러 이름 정규화 ---
    result = df.copy()    # 원본을 보존하기 위해 복사본으로 작업

    # 대소문자와 뒤에 붙은 'scaler' 를 떼어내 이름을 통일한다 ('StandardScaler' -> 'standard')
    name = method.lower().replace('scaler', '').strip()

    # 오타를 냈을 때 KeyError 대신 사용 가능한 이름을 알려준다
    if name not in SCALERS:
        raise ValueError(f"지원하지 않는 스케일러입니다: '{method}' "
                         f"(사용 가능: {list(SCALERS.keys())})")

    # --- 3) 스케일러 학습 및 변환 ---
    # 이름에 해당하는 클래스로 스케일러를 만들어 대상 컬럼의 기준값(평균, 표준편차 등)을 학습시킨다
    scaler = SCALERS[name]()
    result[columns] = scaler.fit_transform(df[columns])

    # --- 4) 변환 내역 출력 (컬럼별 값의 범위 변화) ---
    if verbose:
        print(f'{type(scaler).__name__} 적용 ({len(columns)}개 컬럼)')
        print(f'{"컬럼":12s}{"변환 전":>22s}{"변환 후":>22s}')
        print('-' * 56)

        for c in columns:
            before = f'{df[c].min():.2f} ~ {df[c].max():.2f}'
            after = f'{result[c].min():.2f} ~ {result[c].max():.2f}'
            print(f'{c:12s}{before:>20s}{after:>22s}')

    # --- 5) 학습된 스케일러 저장 (선택) ---
    # 스케일러 안에는 train 에서 구한 기준값(평균·표준편차 등)이 들어 있으므로,
    # 이 파일이 있어야 test 데이터에 똑같은 기준을 적용할 수 있다
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)    # 경로에 없는 폴더가 있으면 만들어 준다

        joblib.dump(scaler, save_path)

        if verbose:
            print(f'\n스케일러 저장: {save_path} ({type(scaler).__name__})')

    # --- 6) 스케일링된 데이터프레임 반환 ---
    return result
