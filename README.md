# OpenLens

OpenLens는 일상적으로 접하는 콘텐츠 속에서 스스로도 의식하지 못한 취향과 관심사를 발견하고, 자신이 세상을 바라보는 방식을 돌아보기 위한 개인용 머신러닝 프로젝트입니다. 단순히 시청 기록을 주제별로 나누는 데 그치지 않고, 어떤 이야기와 관점을 반복해서 접해 왔는지, 무엇을 거의 접하지 않았는지를 살펴봅니다. 궁극적으로는 자신의 정보 소비에 숨어 있는 편향(bias)을 마주하고, 다른 시야를 발견하는 도구를 지향합니다.

## 개발 동기

추천 알고리즘이 보여 주는 콘텐츠를 오래 따라가다 보면 관심사는 더 선명해지지만, 다른 생각을 만날 기회는 줄어들 수 있습니다. 내가 선택한 관심사인지, 반복해서 노출된 결과인지 궁금해진 것이 이 프로젝트의 출발점입니다. 평소에는 알아차리지 못했던 관심사를 데이터에서 찾아내고, 특정 주제나 단체·활동에 시간을 쏟는 동안 다른 의견이나 활동은 얼마나 접했는지 확인하고 싶었습니다.

OpenLens는 개인의 웹·YouTube 이용 기록을 바탕으로 관심사의 구성과 시간에 따른 변화를 살펴보고, 한 주제 안에서 접한 관점의 쏠림과 비어 있는 영역을 드러내려 합니다. 콘텐츠를 소비했다는 사실만으로 그 사람의 신념을 단정하지 않으며, 결과를 스스로 검토하고 수정할 수 있는 방향으로 개발합니다. 실제로 사용할 수 있는 도구를 만들면서 임베딩, 클러스터링, 분류 모델을 직접 실험하고 배우는 것 역시 중요한 목적입니다.

현재 상태: **YouTube·Chrome 기록 읽기와 YouTube 제목의 EmbeddingGemma + K-Means 실험 구현**.

## 첫 완성본의 범위

- 최근 3개월의 웹 방문 기록과 YouTube 시청 기록을 사용자가 직접 가져오기
- 한국어와 영어의 제목·설명·이용 가능한 본문 분석
- 고정된 상위 카테고리와 데이터에서 발견한 세부 관심사를 결합
- 주제별 소비량, 출처 집중도, 유사 콘텐츠 반복, 기간별 관심 변화 표시
- 같은 주제 안에서 자주 접한 논점과 적게 접한 논점 묶기
- 결과를 확인·수정할 수 있는 로컬 대시보드

콘텐츠를 많이 읽었다는 사실만으로 사용자의 신념을 단정하지 않습니다. 분석 결과에는 사용한 기간, 콘텐츠 수, 대표 사례와 불확실성을 함께 표시합니다.

Instagram, 이미지·영상·음성 처리, 자동 수집, 정확한 찬반 판정, 실시간 추천, 다중 사용자 및 외부 배포는 첫 완성본의 범위에 포함하지 않습니다.

## ML 학습 목표

1. `TF-IDF + Logistic Regression`을 상위 주제 분류의 기준 모델로 만듭니다.
2. 다국어 임베딩을 이용한 분류 모델과 기준 모델을 비교합니다.
3. 주제 내부에서 클러스터링으로 세부 관심사와 논점 묶음을 찾습니다.
4. 직접 확인한 소규모 데이터로 분류와 클러스터 결과를 평가합니다.
5. 사용자 수정 결과를 학습 데이터로 축적합니다.

LLM은 선택적으로 클러스터 이름과 주장 후보를 제안할 수 있습니다. 사용자 프로필과 집중도 수치의 계산에는 사용하지 않습니다.

## 개발 흐름

```text
기록 가져오기 → 정제·중복 제거 → 상위 주제 분류
             → 세부 관심사·논점 클러스터링
             → 사용자 확인 → 기간별 분석 → 로컬 대시보드
```

첫 실험은 YouTube 시청 기록 100~300건으로 시작합니다. 제목을 정제하고, EmbeddingGemma 임베딩과 클러스터링으로 어떤 그룹이 나타나는지 확인합니다. 사람이 직접 확인한 약 50건은 초기 평가 자료로 사용합니다. 이 수치는 데이터 확보 상황에 따라 조정합니다.

## 저장소 구조

```text
docs/             범위, 데이터 취급 및 개발 계획
src/openlens/     애플리케이션 코드
tests/            계산·분류 파이프라인 테스트(구현 시 추가)
data/             개인 원본·가공 데이터(추적하지 않음)
models/           학습 결과물(추적하지 않음)
```

YouTube Google Takeout ZIP을 압축 해제하지 않고 읽어 기간별 이벤트 수를 확인할 수 있습니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/openlens /path/to/takeout.zip --days 90
```

원본 ZIP은 저장소 안에 복사하거나 GitHub에 올리지 않습니다. 이 명령은 영상 제목이나 URL을 출력하지 않으며, 분석 결과를 파일로 저장하지도 않습니다. 현재 파서는 YouTube의 HTML 시청 기록 형식을 지원합니다.

여러 YouTube Takeout 내보내기를 하나로 합칠 때는 아래 명령을 사용합니다. 영어·한국어 시청 기록 HTML을 읽고, 동일 영상 ID와 시청 시각이 모두 같은 이벤트만 중복으로 제거합니다. 같은 영상을 다른 시각에 반복 시청했다면 별도 이벤트로 유지합니다. 결과 CSV의 열은 `video_id,title,url,watched_at`이며, 시각은 UTC 오프셋이 포함된 ISO 8601 형식입니다. CSV는 시청 시각순으로 저장됩니다.

```bash
.venv/bin/python -m openlens.merge /path/to/older.zip /path/to/newer.zip --output data/youtube-watch-events.csv
.venv/bin/openlens data/youtube-watch-events.csv --days 90
```

`data/`는 Git에서 제외되지만 CSV에 개인 시청 제목과 URL이 들어 있으므로 공유하지 않습니다. 이후 클러스터링과 시각화 명령에도 ZIP 대신 이 CSV를 입력할 수 있습니다. 이 CSV는 분류 모델의 입력 데이터이며, 사람이 붙일 카테고리 레이블은 아직 포함하지 않습니다.

### 클러스터별 레이블링 샘플

통합 CSV를 클러스터링한 뒤 각 그룹의 영상을 중심 거리 기준 네 구간으로 나눠 20개씩 뽑아 별도 CSV를 만들 수 있습니다. 각 구간에서 5개씩 고르고, 이미 선택한 영상과 임베딩 거리가 먼 후보를 우선합니다. `rank`는 추출 순서이며 중심과 가까운 순위가 아닙니다. 이 파일의 `category_label` 열에 영상별 상위 카테고리를 직접 입력합니다. 같은 클러스터라고 모두 같은 카테고리로 자동 지정하지 않습니다. 기존 수작업 파일은 덮어쓰지 않으므로 새 샘플에는 다른 출력 이름을 지정합니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.label_samples data/youtube-watch-events.csv --k 11 --days 90 --per-cluster 20 --output data/cluster-label-samples-k11-days90-diverse.csv
```

개인 제목이 포함된 결과 파일은 `data/`에만 저장되며, 이미 존재하는 파일은 덮어쓰지 않습니다. `category_label`과 `review_note` 열은 처음에 비어 있습니다.

### 외부 학습 후보와 개인 프로필 분리

부족한 주제의 공개 YouTube 검색 결과를 별도 CSV로 모을 수 있습니다.

```bash
.venv/bin/python -m openlens.external_candidates --output data/external-category-candidates.csv --target 50
```

검색어로 제안한 카테고리는 `label_status=proposed`로 표시됩니다. 영상을 확인해 정답을 확정했다면 해당 행만 `verified`로 바꿉니다. 기본 분류 명령은 `verified` 외부 행만 학습에 사용합니다. 검증 전 후보도 실험에 사용하려면 `--allow-proposed-external`을 명시해야 합니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.classify data/youtube-watch-events.csv data/cluster-label-samples-k10-days90.csv --external-labels data/external-category-candidates.csv --output data/category-predictions-external-verified.csv
```

외부 CSV는 학습 제목과 레이블로만 사용됩니다. 예측 결과와 시청 횟수는 통합된 **개인 시청 기록**에서만 생성되므로, 외부 영상을 개인 관심사·프로필 소비량에 포함하지 않습니다. 외부 검색 결과는 카테고리 정답을 보증하지 않습니다.

개인 검증 레이블과 외부 후보를 **하나의 학습 CSV**로 합칠 수도 있습니다. 원본 두 파일은 그대로 두며, 새 파일의 `source`와 `label_status` 열로 출처와 검증 여부를 구분합니다.

```bash
.venv/bin/python -m openlens.training_data data/cluster-label-samples-k10-days90.csv data/external-category-candidates.csv --output data/training-with-external.csv
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.classify data/youtube-watch-events.csv --training-data data/training-with-external.csv --allow-proposed-external --output data/category-predictions-combined.csv
```

두 번째 명령의 `--allow-proposed-external`은 검증되지 않은 외부 후보까지 학습하는 실험용 옵션입니다. 외부 행을 직접 확인해 `label_status`를 `verified`로 바꾼 뒤에는 해당 옵션 없이 실행할 수 있습니다. 이 학습 CSV는 프로필 데이터 원본이 아닙니다.

외부 후보 전체를 사람이 검증했다면 원본 파일을 유지하면서 검증 완료 사본을 만들 수 있습니다.

```bash
.venv/bin/python -m openlens.training_data data/cluster-label-samples-k10-days90.csv data/external-category-candidates.csv --verify-external --output data/training-with-external-verified.csv
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.classify data/youtube-watch-events.csv --training-data data/training-with-external-verified.csv --output data/category-predictions-verified.csv
```

분류 명령은 예측 CSV와 함께 학습된 로지스틱 회귀 모델을 같은 위치의 `.joblib` 파일로 저장합니다. 위 명령의 모델 파일은 `data/category-predictions-verified.joblib`입니다. `--model-output models/category-classifier.joblib`처럼 저장 경로를 따로 지정할 수도 있습니다. 파일에는 분류기의 학습된 가중치·카테고리와 사용한 임베딩 모델 이름·벡터 차원·scikit-learn 버전이 들어갑니다. EmbeddingGemma 자체나 원본 시청 기록은 저장하지 않습니다. 두 결과 파일 모두 기존 파일을 덮어쓰지 않으며, `data/`와 `models/`는 Git에서 제외합니다. `.joblib` 파일은 신뢰할 수 있는 로컬 파일만 불러와야 하며, 라이브러리 버전을 바꾸면 재학습이 필요할 수 있습니다.

Chrome 기록은 Python 코드에서 `read_chrome_navigation_events("/path/to/Takeout/Chrome")`로 읽을 수 있습니다. 이 함수는 `History.json`의 `Session → tab → navigation`만 처리하고, 시각을 UTC로 바꾼 뒤 과거순으로 정렬합니다. `Browser History`와 `Typed Url`은 아직 처리하지 않습니다. 세션 이동 기록은 전체 방문 기록이 아니며, 다른 Chrome Takeout 파일(주소·기기 설정 등)은 읽지 않습니다.

## 첫 클러스터링 실험

```bash
.venv/bin/python -m openlens.cluster data/youtube-watch-events.csv --k 11 --days 90
```

`--k`는 실험할 그룹 수이며 기본값은 11입니다. 이는 확정된 카테고리 수가 아니라 비교 실험에서 선택한 값입니다. EmbeddingGemma가 최근 90일의 고유 영상 제목을 클러스터링용 벡터로 바꾸고 K-Means가 그룹 번호를 붙입니다. 같은 영상의 반복 시청은 그룹을 만드는 데 중복 입력하지 않고, 결과의 시청 횟수에만 합산합니다. 그룹별 대표 제목은 사람이 임시로 해석해야 하며, 자동으로 붙인 주제 이름이 아닙니다. 결과는 터미널에만 출력되고 저장되지 않습니다. 대표 제목은 개인 시청 정보를 드러낼 수 있으므로 출력 화면 공유에 주의합니다.

모델 `google/embeddinggemma-300m`을 처음 사용할 때는 Hugging Face 계정에서 [Gemma 이용 약관](https://huggingface.co/google/embeddinggemma-300m)에 동의하고 로컬 환경에서 인증해야 합니다. 최초 실행 시 모델 파일을 다운로드하며, 이후 제목 임베딩은 로컬에서 계산합니다. 모델은 `float16` 활성값을 지원하지 않으므로 기본 `float32` 설정을 사용합니다.

### 로컬 시각화

```bash
.venv/bin/python -m openlens.visualize data/youtube-watch-events.csv --k 11 --days 90
```

이 명령은 `artifacts/clusters-k11-days90.html`에 오프라인 HTML 보고서를 만듭니다. 파일을 브라우저로 열면 클러스터별 고유 영상 수·시청 횟수 막대그래프와 제목별 점 분포도를 볼 수 있습니다. 점의 크기는 시청 횟수이고, 마우스를 올리면 제목이 나타납니다. 점의 위치는 K-Means에 사용한 동일한 임베딩 벡터를 표시용으로 2차원 PCA 투영한 것이므로 원래 고차원 거리와 정확히 같지는 않습니다. 보고서에는 개인 시청 제목이 포함되므로 `artifacts/`는 Git에서 제외합니다. 그룹을 직접 수정하는 기능은 아직 없습니다. 다른 `--k` 값으로 다시 실행해 별도 보고서를 비교할 수 있습니다.

## 상위 카테고리 안의 하위 주제 실험

상위 분류 결과에서 카테고리 하나를 골라 다시 군집화할 수 있습니다. Entertainment를 `k=5`로 묶고, 각 군집을 중심 거리 순서로 네 구간으로 나눠 구간마다 5개씩 레이블링 샘플로 저장합니다. 각 구간 안에서는 이미 고른 영상과 임베딩 거리가 먼 영상을 우선합니다. 상위 카테고리가 `review_needed`였던 영상도 탐색에는 포함하지만, 샘플 CSV에 그 상태를 표시합니다. 이 상태를 하위 주제의 정답으로 사용하지는 않습니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.subcategory sample data/category-predictions-verified-saved.csv --parent Entertainment --k 5 --per-cluster 20
```

전체 군집 결과는 `data/subcategories/entertainment-clusters-k5-diverse.csv`, 사람이 입력할 샘플은 `data/subcategories/entertainment-label-samples-k5-diverse.csv`입니다. 기존 `-k5.csv` 레이블 파일은 그대로 남습니다. 샘플의 `primary_label`에 주 하위 카테고리를 하나 입력하고, 필요한 경우 `secondary_labels`에 보조 카테고리를 `|`로 구분해 입력합니다. 해당 상위 카테고리가 아니거나 제목만으로 판단하기 어렵다면 `primary_label`을 비워 두고 `review_note`에 이유를 적습니다. 허용되는 하위 카테고리 후보는 [분류표](src/openlens/subcategory_taxonomy.py)에서 수정할 수 있습니다. 이 후보는 확정된 카테고리가 아니며, `cluster_id`도 주제 이름이 아닙니다.

두 종류 이상의 주 레이블을 직접 입력한 뒤에만 아래 분류 명령을 실행할 수 있습니다. 이 명령은 **사람이 입력한 주 레이블만** 학습해 Entertainment의 나머지 영상에 적용합니다. 보조 레이블은 원본 입력 그대로 보존하되 아직 자동 예측하지 않습니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.subcategory classify data/subcategories/entertainment-clusters-k5-diverse.csv data/subcategories/entertainment-label-samples-k5-diverse.csv --parent Entertainment --output data/subcategories/entertainment-predictions-diverse.csv --model-output models/entertainment-subcategory-diverse.joblib
```

다른 상위 카테고리도 `--parent`를 바꿔 같은 절차를 반복합니다. 각 카테고리의 데이터가 적으면 `k=5`가 의미 없거나 실행되지 않을 수 있으므로 그때는 `--k`를 낮추거나 상위 레이블을 먼저 보강합니다. `data/`와 `models/`의 개인 결과물은 Git에서 제외합니다.

## 개인정보와 GitHub

방문 기록, 시청 기록, 개인 프로필, 토큰, 학습된 개인 모델은 로컬에 보관합니다. 이 저장소에는 코드, 설계 문서, 실험 설정·집계 결과 및 개인정보가 없는 예시만 올립니다. Git에 추가하기 전에는 `git status`와 변경 내용을 확인합니다. 자세한 기준은 [데이터 취급 원칙](docs/data-policy.md)을 참고합니다.

## 로드맵

| 단계 | 결과물 |
| --- | --- |
| 1 | YouTube 기록 샘플과 첫 클러스터링 실험 |
| 2 | 웹·YouTube 기록 파서와 공통 데이터 형식 |
| 3 | 상위 주제 분류 모델 및 세부 관심사 발견 |
| 4 | 논점 클러스터와 기간별 집중도·변화 분석 |
| 5 | 수정 가능한 로컬 대시보드와 평가 보고서 |

세부 기준은 [프로젝트 계획](docs/project-plan.md)에 정리했습니다.
