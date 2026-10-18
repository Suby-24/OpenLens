# OpenLens

OpenLens는 내가 소비한 콘텐츠를 바탕으로 관심사와 정보 소비 패턴을 살펴보는 개인용 ML 프로젝트입니다. 웹 방문 기록과 YouTube 시청 기록에서 주제를 찾고, 한 주제 안에서 반복해서 접하는 논점과 시간에 따른 변화를 분석하는 것을 목표로 합니다.

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

원본 ZIP을 저장소 안에 복사하거나 GitHub에 올리지 마세요. 이 명령은 영상 제목이나 URL을 출력하지 않으며, 분석 결과를 파일로 저장하지도 않습니다. 현재 파서는 YouTube의 HTML 시청 기록 형식을 지원합니다.

여러 YouTube Takeout 내보내기를 하나로 합칠 때는 아래처럼 실행합니다. 영어·한국어 시청 기록 HTML을 읽고, 동일 영상 ID와 시청 시각이 모두 같은 이벤트만 중복으로 제거합니다. 같은 영상을 다른 시각에 반복 시청했다면 별도 이벤트로 유지합니다. 결과 CSV의 열은 `video_id,title,url,watched_at`이며, 시각은 UTC 오프셋이 포함된 ISO 8601 형식입니다. CSV는 시청 시각순으로 저장됩니다.

```bash
.venv/bin/python -m openlens.merge /path/to/older.zip /path/to/newer.zip --output data/youtube-watch-events.csv
.venv/bin/openlens data/youtube-watch-events.csv --days 90
```

`data/`는 Git에서 제외되지만 CSV에 개인 시청 제목과 URL이 들어 있으므로 공유하지 마세요. 이후 클러스터링과 시각화 명령에도 ZIP 대신 이 CSV를 입력할 수 있습니다. 이 CSV는 분류 모델의 입력 데이터이며, 사람이 붙일 카테고리 레이블은 아직 포함하지 않습니다.

### 클러스터별 레이블링 샘플

통합 CSV를 클러스터링한 뒤 각 그룹에서 중심에 가장 가까운 영상 20개씩 뽑아 별도 CSV를 만들 수 있습니다. `rank=1`이 그룹 중심에 가장 가까운 영상입니다. 이 파일의 `category_label` 열에 영상별 상위 카테고리를 직접 입력하세요. 같은 클러스터라고 모두 같은 카테고리로 자동 지정하지 않습니다. 이 샘플은 대표 사례를 확인하기 위한 것이며, 그룹 내부의 예외까지 평가하려면 나중에 무작위 샘플도 추가해야 합니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.label_samples data/youtube-watch-events.csv --k 10 --days 90 --per-cluster 20 --output data/cluster-label-samples-k10-days90.csv
```

개인 제목이 포함된 결과 파일은 `data/`에만 저장되며, 이미 존재하는 파일은 덮어쓰지 않습니다. `category_label`과 `review_note` 열은 처음에 비어 있습니다.

### 외부 학습 후보와 개인 프로필 분리

부족한 주제의 공개 YouTube 검색 결과를 별도 CSV로 모을 수 있습니다.

```bash
.venv/bin/python -m openlens.external_candidates --output data/external-category-candidates.csv --target 50
```

검색어로 제안한 카테고리는 `label_status=proposed`로 표시됩니다. 영상을 확인해 정답을 확정했다면 해당 행만 `verified`로 바꾸세요. 기본 분류 명령은 `verified` 외부 행만 학습에 사용합니다. 검증 전 후보도 실험에 사용하려면 `--allow-proposed-external`을 명시해야 합니다.

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

Chrome 기록은 Python 코드에서 `read_chrome_navigation_events("/path/to/Takeout/Chrome")`로 읽을 수 있습니다. 이 함수는 `History.json`의 `Session → tab → navigation`만 처리하고, 시각을 UTC로 바꾼 뒤 과거순으로 정렬합니다. `Browser History`와 `Typed Url`은 아직 처리하지 않습니다. 세션 이동 기록은 전체 방문 기록이 아니며, 다른 Chrome Takeout 파일(주소·기기 설정 등)은 읽지 않습니다.

## 첫 클러스터링 실험

```bash
.venv/bin/python -m openlens.cluster data/youtube-watch-events.csv --k 8 --days 90
```

`--k`는 실험할 그룹 수이며 8은 실행 예시일 뿐 확정된 카테고리 수가 아닙니다. EmbeddingGemma가 최근 90일의 고유 영상 제목을 클러스터링용 벡터로 바꾸고 K-Means가 그룹 번호를 붙입니다. 같은 영상의 반복 시청은 그룹을 만드는 데 중복 입력하지 않고, 결과의 시청 횟수에만 합산합니다. 그룹별 대표 제목은 사람이 임시로 해석해야 하며, 자동으로 붙인 주제 이름이 아닙니다. 결과는 터미널에만 출력되고 저장되지 않습니다. 대표 제목은 개인 시청 정보를 드러낼 수 있으니 출력 화면을 공유할 때 주의하세요.

모델 `google/embeddinggemma-300m`을 처음 사용할 때는 Hugging Face 계정에서 [Gemma 이용 약관](https://huggingface.co/google/embeddinggemma-300m)에 동의하고 로컬 환경에서 인증해야 합니다. 최초 실행 시 모델 파일을 다운로드하며, 이후 제목 임베딩은 로컬에서 계산합니다. 모델은 `float16` 활성값을 지원하지 않으므로 기본 `float32` 설정을 사용합니다.

### 로컬 시각화

```bash
.venv/bin/python -m openlens.visualize data/youtube-watch-events.csv --k 8 --days 90
```

이 명령은 `artifacts/clusters-k8-days90.html`에 오프라인 HTML 보고서를 만듭니다. 파일을 브라우저로 열면 클러스터별 고유 영상 수·시청 횟수 막대그래프와 제목별 점 분포도를 볼 수 있습니다. 점의 크기는 시청 횟수이고, 마우스를 올리면 제목이 나타납니다. 점의 위치는 K-Means에 사용한 동일한 임베딩 벡터를 표시용으로 2차원 PCA 투영한 것이므로 원래 고차원 거리와 정확히 같지는 않습니다. 보고서에는 개인 시청 제목이 포함되므로 `artifacts/`는 Git에서 제외합니다. 그룹을 직접 수정하는 기능은 아직 없습니다. 다른 `--k` 값으로 다시 실행해 별도 보고서를 비교할 수 있습니다.

## 개인정보와 GitHub

방문 기록, 시청 기록, 개인 프로필, 토큰, 학습된 개인 모델은 로컬에 보관합니다. 이 저장소에는 코드, 설계 문서, 실험 설정·집계 결과 및 개인정보가 없는 예시만 올립니다. Git에 추가하기 전에 `git status`와 변경 내용을 확인하세요. 자세한 기준은 [데이터 취급 원칙](docs/data-policy.md)을 참고하세요.

## 로드맵

| 단계 | 결과물 |
| --- | --- |
| 1 | YouTube 기록 샘플과 첫 클러스터링 실험 |
| 2 | 웹·YouTube 기록 파서와 공통 데이터 형식 |
| 3 | 상위 주제 분류 모델 및 세부 관심사 발견 |
| 4 | 논점 클러스터와 기간별 집중도·변화 분석 |
| 5 | 수정 가능한 로컬 대시보드와 평가 보고서 |

세부 기준은 [프로젝트 계획](docs/project-plan.md)에 정리했습니다.
