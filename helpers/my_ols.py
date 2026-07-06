from pandas import DataFrame
from statsmodels.api import add_constant, OLS

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