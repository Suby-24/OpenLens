# OpenLens

OpenLens는 내가 평소에 보는 콘텐츠를 통해 **미처 알아차리지 못했던 관심사와 편향**을 살펴보려는 개인 머신러닝 프로젝트입니다. 어떤 주제와 의견을 반복해서 접했는지, 반대로 무엇은 거의 보지 않았는지를 알아보고 싶었습니다. 궁극적으로는 내 편향을 직시하고, 그와 다른 의견을 만날 수 있는 도구를 만드는 것이 목표입니다.

## 개발 동기

어느 순간부터 내가 보는 콘텐츠가 비슷해지고 있다는 생각이 들었습니다. 관심 있는 주제를 계속 보게 되는 건 자연스럽지만, 그러다 보면 나와 다른 의견은 얼마나 접하고 있는지 알기 어렵습니다. 내가 스스로 골라 보고 있는 건지, 추천 알고리즘이 보여 주는 것을 익숙해서 계속 보고 있는 건지도 궁금했습니다.

특히 **내가 미처 알아차리지 못한 편향을 보고 싶었습니다.** 어떤 주제에 많은 시간을 쓰면서도 그 안에서 한쪽 의견만 반복해서 접하고 있지는 않을까? 반대되는 주장이나 활동이 있다는 사실조차 모르고 지나치지는 않을까? 내 시청 기록과 방문 기록을 살펴보면 이런 질문에 조금은 답할 수 있을 것 같았습니다.

그래서 OpenLens를 시작했습니다. 내가 무엇에 관심을 두고 있는지, 어떤 관점을 자주 만나고 무엇을 거의 보지 않았는지 보여 주는 도구를 만들고 싶습니다. 궁극적으로는 **나도 몰랐던 나의 편향을 직시하고, 그와 다른 의견을 만날 수 있도록 돕는 것**이 목표입니다. 콘텐츠 기록만으로 내 생각을 단정하기보다는, 결과를 내가 직접 확인하고 돌아볼 수 있게 만들려고 합니다.

이 과정을 직접 구현하면서 임베딩, 클러스터링, 분류 모델을 배우고 실험하는 것도 이 프로젝트를 시작한 중요한 이유입니다.

현재는 **YouTube·Chrome 기록을 읽고, YouTube 영상 제목을 임베딩해 군집화·분류하는 단계**까지 구현했습니다. 편향과 반대 의견을 분석하는 기능은 앞으로 만들어 갈 부분입니다.

## 첫 완성본의 범위

- 최근 3개월의 웹 방문 기록과 YouTube 시청 기록을 사용자가 직접 가져오기
- 한국어와 영어의 제목·설명·이용 가능한 본문 분석
- 고정된 상위 카테고리와 데이터에서 발견한 세부 관심사를 결합
- 주제별 소비량, 출처 집중도, 유사 콘텐츠 반복, 기간별 관심 변화 표시
- 같은 주제 안에서 자주 접한 논점과 적게 접한 논점 묶기
- 결과를 확인·수정할 수 있는 로컬 대시보드

어떤 콘텐츠를 많이 봤다고 해서 그 의견에 동의한다고 단정할 수는 없습니다. 분석 결과를 보여줄 때도 기간과 콘텐츠 수, 대표 사례, 판단이 불확실한 부분을 함께 드러내려 합니다.

처음부터 모든 기능을 넣으려 하지는 않습니다. Instagram, 이미지·영상·음성 처리, 자동 수집, 정확한 찬반 판정, 실시간 추천, 다중 사용자와 외부 배포는 첫 완성본 이후에 다시 생각해 볼 계획입니다.

## ML 학습 목표

1. `TF-IDF + Logistic Regression`을 상위 주제 분류의 기준 모델로 만듭니다.
2. 다국어 임베딩을 이용한 분류 모델과 기준 모델을 비교합니다.
3. 주제 내부에서 클러스터링으로 세부 관심사와 논점 묶음을 찾습니다.
4. 직접 확인한 소규모 데이터로 분류와 클러스터 결과를 평가합니다.
5. 사용자 수정 결과를 학습 데이터로 축적합니다.

LLM은 나중에 클러스터 이름이나 주장 후보를 제안하는 데 활용할 수 있습니다. 관심사와 소비량을 계산하는 핵심 과정은 직접 만든 데이터 처리·ML 흐름으로 살펴보려 합니다.

## 개발 흐름

```text
기록 가져오기 → 정제·중복 제거 → 상위 주제 분류
             → 세부 관심사·논점 클러스터링
             → 사용자 확인 → 기간별 분석 → 로컬 대시보드
```

처음에는 YouTube 시청 기록의 제목을 정리한 뒤, EmbeddingGemma 임베딩과 클러스터링으로 어떤 그룹이 나오는지 살펴봤습니다. 일부 영상은 직접 확인하고 레이블을 붙여 분류 모델의 학습·평가에 사용합니다. 샘플 수와 범위는 데이터가 쌓이는 만큼 조정할 예정입니다.

## 저장소 구조

```text
docs/             범위, 데이터 취급 및 개발 계획
src/openlens/     애플리케이션 코드
tests/            데이터 처리·분류 코드의 동작 테스트
data/             개인 원본·가공 데이터(추적하지 않음)
models/           학습 결과물(추적하지 않음)
```

먼저 Google Takeout으로 받은 YouTube ZIP에서 시청 기록이 얼마나 들어 있는지 확인할 수 있습니다. ZIP을 미리 풀 필요는 없습니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/openlens /path/to/takeout.zip --days 90
```

원본 ZIP에는 개인 기록이 있으므로 저장소에 복사하거나 GitHub에 올리지 않습니다. 위 명령은 영상 제목과 URL 대신 건수만 보여주고, 결과 파일도 만들지 않습니다. 현재는 YouTube의 HTML 시청 기록을 읽을 수 있습니다.

Takeout을 여러 번 받았다면 아래 명령으로 하나로 합칠 수 있습니다. 영어·한국어 시청 기록을 읽고, 영상 ID와 시청 시각이 **둘 다 같은 기록만** 중복으로 제거합니다. 같은 영상을 다른 때에 다시 봤다면 별도 시청으로 남깁니다. 결과 CSV는 `video_id,title,url,watched_at` 열을 가지며, 시각은 UTC 오프셋을 포함한 ISO 8601 형식으로 시청 시각순 저장됩니다.

```bash
.venv/bin/python -m openlens.merge /path/to/older.zip /path/to/newer.zip --output data/youtube-watch-events.csv
.venv/bin/openlens data/youtube-watch-events.csv --days 90
```

이 CSV에는 내가 본 영상의 제목과 URL이 들어 있습니다. `data/`는 Git에서 제외했지만 파일 자체도 공유하지 않도록 주의해야 합니다. 이후 군집화·시각화에는 ZIP 대신 이 CSV를 사용할 수 있습니다. 아직 사람이 붙인 카테고리 레이블은 없는, 시청 기록 원본에 가까운 파일입니다.

### 클러스터별 레이블링 샘플

모든 영상을 하나씩 레이블링하기는 어렵습니다. 그래서 먼저 군집을 만들고, 각 그룹에서 사람이 살펴볼 영상 20개를 뽑습니다. 중심에 가까운 영상만 고르면 비슷한 제목이 몰릴 수 있어, 중심부터 바깥쪽까지 네 구간에서 5개씩 고릅니다. 각 구간 안에서도 이미 고른 영상과 임베딩 거리가 먼 후보를 우선합니다. `rank`는 뽑힌 순서일 뿐 중심과의 거리 순위는 아닙니다. 영상별 정답은 `category_label`에 직접 입력합니다. 같은 군집에 있다고 모두 같은 카테고리로 지정하지는 않습니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.label_samples data/youtube-watch-events.csv --k 11 --days 90 --per-cluster 20 --output data/cluster-label-samples-k11-days90-diverse.csv
```

처음에는 `category_label`과 `review_note`가 비어 있습니다. 직접 채운 내용을 잃지 않도록 기존 파일은 덮어쓰지 않으니, 새 실험을 할 때는 `--output`에 다른 이름을 지정해야 합니다. 결과 파일은 개인 제목이 들어 있는 `data/`에 보관합니다.

### 외부 학습 후보와 개인 프로필 분리

내 기록만으로 학습 예시가 부족한 주제는 공개 YouTube 검색 결과에서 후보를 모을 수 있습니다.

```bash
.venv/bin/python -m openlens.external_candidates --output data/external-category-candidates.csv --target 50
```

검색어로 찾은 영상의 카테고리가 모두 맞는 것은 아닙니다. 처음에는 `label_status=proposed`로 두고, 내용을 확인해 정답이라고 판단한 행만 `verified`로 바꿉니다. 기본 분류 명령은 이렇게 확인한 외부 영상만 학습합니다. 확인하지 않은 후보까지 시험해 보고 싶을 때에만 `--allow-proposed-external`을 붙입니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.classify data/youtube-watch-events.csv data/cluster-label-samples-k10-days90.csv --external-labels data/external-category-candidates.csv --output data/category-predictions-external-verified.csv
```

외부 영상은 분류 모델을 가르치는 예시로만 사용합니다. **내 관심사와 시청 횟수**는 내 시청 기록에서만 계산하며, 검색으로 찾은 영상을 내가 본 것처럼 섞지 않습니다.

직접 확인한 개인 영상과 외부 후보를 하나의 학습 CSV로 합칠 수도 있습니다. 원본은 그대로 두고, 새 파일의 `source`와 `label_status`로 어디에서 왔고 검증했는지를 구분합니다.

```bash
.venv/bin/python -m openlens.training_data data/cluster-label-samples-k10-days90.csv data/external-category-candidates.csv --output data/training-with-external.csv
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.classify data/youtube-watch-events.csv --training-data data/training-with-external.csv --allow-proposed-external --output data/category-predictions-combined.csv
```

두 번째 명령의 `--allow-proposed-external`은 아직 확인하지 않은 외부 후보까지 학습하는 실험용 옵션입니다. 외부 행을 검증해 `verified`로 바꿨다면 이 옵션 없이 실행할 수 있습니다. 합쳐진 CSV는 학습용이지, 개인 프로필을 계산하는 시청 기록은 아닙니다.

외부 후보를 전부 직접 확인했다면, 원본은 남겨 두고 검증 완료 사본을 만들 수 있습니다.

```bash
.venv/bin/python -m openlens.training_data data/cluster-label-samples-k10-days90.csv data/external-category-candidates.csv --verify-external --output data/training-with-external-verified.csv
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.classify data/youtube-watch-events.csv --training-data data/training-with-external-verified.csv --output data/category-predictions-verified.csv
```

분류를 실행하면 예측 CSV와 학습된 로지스틱 회귀 모델이 함께 저장됩니다. 위 명령에서는 모델이 `data/category-predictions-verified.joblib`에 저장되며, `--model-output models/category-classifier.joblib`처럼 경로를 따로 정할 수도 있습니다. 모델 파일에는 학습된 가중치와 카테고리, 사용한 임베딩 모델 이름·벡터 차원·scikit-learn 버전이 들어갑니다. EmbeddingGemma 자체나 원본 시청 기록을 복사해 넣지는 않습니다. 기존 파일은 덮어쓰지 않고, `data/`와 `models/`도 Git에 올리지 않습니다. `.joblib`은 신뢰할 수 있는 로컬 파일만 불러와야 하며, 라이브러리 버전이 달라지면 재학습해야 할 수 있습니다.

Chrome 기록은 Python에서 `read_chrome_navigation_events("/path/to/Takeout/Chrome")`로 읽을 수 있습니다. 지금은 `History.json`의 `Session → tab → navigation`만 처리하고, 시각을 UTC로 바꿔 과거순으로 정렬합니다. `Browser History`와 `Typed Url`은 아직 다루지 않으므로, 이 결과를 전체 방문 기록으로 받아들이면 안 됩니다. 주소나 기기 설정 같은 다른 Chrome Takeout 파일도 읽지 않습니다.

## 첫 클러스터링 실험

```bash
.venv/bin/python -m openlens.cluster data/youtube-watch-events.csv --k 11 --days 90
```

`--k`는 몇 개의 그룹을 만들어 볼지 정하는 값입니다. 기본값 11은 비교 실험에서 선택한 값이지, 확정된 카테고리 개수가 아닙니다. EmbeddingGemma가 최근 90일의 고유 영상 제목을 벡터로 바꾸고 K-Means가 비슷한 제목을 묶습니다. 같은 영상을 여러 번 봤다면 군집에는 한 번만 넣고, 시청 횟수에는 반복을 반영합니다. 그룹의 대표 제목을 보고 무엇에 관한 군집인지 해석하는 일은 아직 사람의 몫입니다. 결과는 터미널에만 나오며 파일로 저장되지 않습니다. 화면에 개인 시청 제목이 보일 수 있으니 공유할 때 주의해야 합니다.

`google/embeddinggemma-300m`을 처음 쓸 때는 Hugging Face 계정에서 [Gemma 이용 약관](https://huggingface.co/google/embeddinggemma-300m)에 동의하고 로컬 환경에서 인증해야 합니다. 모델 파일을 한 번 다운로드한 뒤에는 제목 임베딩을 로컬에서 계산합니다. 이 모델은 `float16` 활성값을 지원하지 않아 기본 `float32` 설정을 사용합니다.

### 로컬 시각화

```bash
.venv/bin/python -m openlens.visualize data/youtube-watch-events.csv --k 11 --days 90
```

이 명령은 `artifacts/clusters-k11-days90.html`에 오프라인 보고서를 만듭니다. 브라우저로 열면 군집별 고유 영상 수와 시청 횟수, 제목의 점 분포를 볼 수 있습니다. 점이 클수록 많이 본 영상이고, 마우스를 올리면 제목이 나옵니다. 화면에 표시하기 위해 임베딩 벡터를 PCA로 2차원에 줄였기 때문에, 그래프에서 보이는 거리가 원래의 고차원 거리를 그대로 나타내지는 않습니다. 개인 시청 제목이 담긴 보고서는 `artifacts/`에 두고 Git에서는 제외합니다. 아직 화면에서 그룹을 직접 고칠 수는 없지만, 다른 `--k` 값으로 보고서를 다시 만들어 비교할 수 있습니다.

## 상위 카테고리 안의 하위 주제 실험

상위 카테고리만으로는 내가 무엇을 주로 봤는지 충분히 알기 어렵습니다. 그래서 카테고리 하나를 골라 그 안에서 다시 비슷한 영상을 찾아봅니다. 아래 예시는 Entertainment를 `k=5`로 묶고, 각 군집의 중심부터 바깥쪽까지 네 구간에서 영상 5개씩을 뽑습니다. 한 구간 안에서도 이미 고른 영상과 다른 후보를 우선합니다. 상위 분류가 `review_needed`인 영상도 탐색에는 포함하되, 샘플 CSV에 상태를 남깁니다. 그 상태를 하위 주제의 정답으로 쓰지는 않습니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.subcategory sample data/category-predictions-verified-saved.csv --parent Entertainment --k 5 --per-cluster 20
```

전체 군집 결과는 `data/subcategories/entertainment-clusters-k5-diverse.csv`, 직접 레이블링할 샘플은 `data/subcategories/entertainment-label-samples-k5-diverse.csv`에 저장됩니다. 기존 `-k5.csv` 레이블 파일은 그대로 둡니다. 샘플의 `primary_label`에 가장 잘 맞는 하위 카테고리 하나를 적고, 보조 카테고리가 필요하다면 `secondary_labels`에 `|`로 구분해 적습니다. 제목만으로 모르겠거나 Entertainment에 속하지 않는 영상이라면 주 레이블을 비워 두고 `review_note`에 이유를 남기면 됩니다. 쓸 수 있는 이름은 [하위 카테고리 초안](src/openlens/subcategory_taxonomy.py)에서 확인하고 바꿀 수 있습니다. `cluster_id`는 임시 그룹 번호이지 카테고리 이름이 아닙니다.

서로 다른 주 레이블을 두 종류 이상 입력해야 아래 명령으로 학습할 수 있습니다. 모델은 **직접 확인한 주 레이블만** 배우고, Entertainment의 나머지 영상에는 하위 카테고리를 제안합니다. 보조 레이블은 입력한 내용을 보존하지만 아직 자동으로 예측하지는 않습니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m openlens.subcategory classify data/subcategories/entertainment-clusters-k5-diverse.csv data/subcategories/entertainment-label-samples-k5-diverse.csv --parent Entertainment --output data/subcategories/entertainment-predictions-diverse.csv --model-output models/entertainment-subcategory-diverse.joblib
```

다른 상위 카테고리도 `--parent`를 바꾸면 같은 방법으로 살펴볼 수 있습니다. 영상이 적은 카테고리에서는 `k=5`가 의미 없거나 실행되지 않을 수 있습니다. 그럴 때는 `--k`를 낮추거나 상위 레이블을 먼저 보강해야 합니다. 개인 결과물은 계속 `data/`와 `models/`에만 보관합니다.

## 개인정보와 GitHub

이 프로젝트를 만들면서 가장 조심해야 할 것은 개인 기록입니다. 방문·시청 기록과 프로필, 토큰, 개인 데이터로 학습한 모델은 로컬에만 둡니다. GitHub에는 코드와 문서, 개인정보가 없는 예시만 올리려 합니다. 새 파일을 Git에 추가하기 전에는 `git status`와 변경 내용을 확인합니다. 기준은 [데이터 취급 원칙](docs/data-policy.md)에 정리했습니다.

## 로드맵

| 단계 | 결과물 |
| --- | --- |
| 1 | YouTube 기록 샘플과 첫 클러스터링 실험 |
| 2 | 웹·YouTube 기록 파서와 공통 데이터 형식 |
| 3 | 상위 주제 분류 모델 및 세부 관심사 발견 |
| 4 | 논점 클러스터와 기간별 집중도·변화 분석 |
| 5 | 수정 가능한 로컬 대시보드와 평가 보고서 |

앞으로 만들 기능과 범위는 [프로젝트 계획](docs/project-plan.md)에 조금 더 자세히 적어 두었습니다.
