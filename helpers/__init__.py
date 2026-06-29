import os
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from pathlib import Path

# -----------------------------------
# 내보낼 모듈 임포트
# -----------------------------------
from . import my_plot        # 데이터 시각화 관련 함수 모듈
from . import my_qtcheck     # 데이터 품질 점검 관련 함수 모듈
from . import my_stats       # 통계 분석 관련 함수 모듈
from . import my_prep        # 데이터 전처리 관련 함수 모듈

# -----------------------------------
# 한글 폰트 설정
# -----------------------------------
fpath = "./helpers/fonts"  # 한글을 지원하는 폰트 파일의 경로
font_files = os.listdir(fpath) # 폰트 파일이 있는지 확인

for f in font_files:
    font_path=os.path.join(fpath, f)      # 폰트 파일의 전체 경로
    fm.fontManager.addfont(font_path)     # 폰트 등록
    fprop=fm.FontProperties(fname=font_path) # 폰트의 속성을 읽어옴
    fname = fprop.get_name()              # 읽어온 속성에서 폰트의 이름만 추출
    plt.rcParams['font.family']=fname     # 그래프에 한글 폰트 적용


# -----------------------------------
# 그래프 기본 설정
# -----------------------------------

my_dpi = 200                                # 이미지 선명도(100~300)
plt.rcParams['font.size'] = 12              # 기본 폰트 크기
plt.rcParams['axes.unicode_minus'] = False  # 그래프에 마이너스 깨짐 방지
plt.rcParams['figure.dpi'] = my_dpi         # 그래프의 dpi 설정
plt.rcParams['savefig.dpi'] = my_dpi        # 저장되는 그래프의 dpi 설정
plt.rcParams['lines.linewidth'] =2          # 그래프의 선 굵기 설정
plt.rcParams['axes.axisbelow'] = True       # 그래프의 축과 격자선을 선보다 뒤에 배치