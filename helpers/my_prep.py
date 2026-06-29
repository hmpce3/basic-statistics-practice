from pandas import pivot_table
from . import my_stats

def long2wide(df, hue, values, dropna=True):
    """
    long format 데이터프레임을 group 단위로 컬럼을 펼처 wide format으로 변환하는 함수

    Args:
        - df : 변환할 데이터 프레임
        - hue : 펼칠 기준이 되는 그룹 열 이름(각 값이 새 열이 됨)
        - values : 펼칠 값이 담긴 열 이름
        - dropna : 결측치 행을 결과에서 제외할 지 여부 (기본값 : True)

    Returns
        - wide format으로 변환된 데이터프레임
    """
    wide = pivot_table(data=df,
                       index=df.groupby(hue).cumcount(),
                       columns=hue, values=values, dropna=dropna)
    
    wide.columns.name = None
    wide.index.name = None
    return wide

#==================================
# 다중 공선성 제거
#==================================
from pandas import pivot_table
from . import my_stats


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

    # 대상에서 제외되는 컬럼(종속변수, 명목형 등)은 그대로 보존하기 위해 컬럼이름을 따로 기록
    keep = []
    for c in df.columns:
        if c not in targets:
            keep.append(c)

    # --- 2) 반복 제거 과정 ---
    work = df[targets].copy()       # 원본을 보존하기 위해 대상 변수만 복사본으로 작업
    step = 0                        # 반복 단계 카운터

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