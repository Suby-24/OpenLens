"""Train a category classifier from reviewed YouTube titles."""

from __future__ import annotations

import argparse # Easy command-line argument parsing
import csv
from collections import Counter # 특정한 리스트에 count를 해서 딕셔너리 형태로 변환
from pathlib import Path # 디렉토리 관련 모듈

import joblib # 머신러닝 파일을 저장하고 다시 불러올때 사용
import numpy as np
import sklearn #머신러닝 모듈이 저장됨
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .cluster import EMBEDDING_MODEL, embed_titles
from .takeout import read_watch_events


def read_labels(path: Path) -> dict[str, str]:
    # Load reviewed personal labels and reject conflicting labels for one video.
    labels: dict[str, str] = {} # 레이블 변수 초기화
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        # 비디오 아이디랑 카테고리 레이블 필드가 없으면 에러 발생
        if not reader.fieldnames or not {"video_id", "category_label"} <= set(reader.fieldnames):
            raise ValueError("Label CSV needs video_id and category_label columns")
        for row in reader:
            video_id, label = row["video_id"].strip(), row["category_label"].strip() # 잘못된 레이블 차단
            if not video_id or not label:
                continue
            if video_id in labels and labels[video_id] != label: # 같은 아이디에 이미 저장된 레이블이 있는데 다른 레이블이 나오면 충돌 에러로 판단
                raise ValueError(f"Conflicting labels for video_id: {video_id}")
            labels[video_id] = label
    return labels


# 외부 합성 데이터를 읽어올때 사용함
def read_external_labels(path: Path, *, allow_proposed: bool = False) -> list[tuple[str, str, str]]:
    """Read external titles for training only; never use them as watch events."""
    # Keep verified candidates unless the caller explicitly allows proposed ones.
    rows: list[tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"video_id", "title", "category_label", "label_status"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("External CSV needs video_id,title,category_label,label_status")
        for row in reader:
            status = row["label_status"].strip()
            if status == "verified" or (allow_proposed and status == "proposed"):
                if row["video_id"].strip() and row["title"].strip() and row["category_label"].strip():
                    rows.append((row["video_id"].strip(), row["title"].strip(), row["category_label"].strip()))
    return rows


def read_training_data(path: Path, *, allow_proposed: bool = False) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Read one training file while keeping personal and external origins distinct."""
    # Separate personal labels from external examples before model training.
    personal: dict[str, str] = {}
    external: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"video_id", "title", "category_label", "source", "label_status"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("Training CSV needs video_id,title,category_label,source,label_status")
        for row in reader:
            video_id = row["video_id"].strip()
            label = row["category_label"].strip()
            if not video_id or not label:
                continue
            if video_id in seen:
                raise ValueError(f"Duplicate training video ID: {video_id}")
            seen.add(video_id)
            source = row["source"].strip()
            status = row["label_status"].strip()
            # 개인 데이터와 외부 데이터를 따로 구분함
            if source == "personal" and status == "verified":
                personal[video_id] = label
            elif source == "external" and (status == "verified" or (allow_proposed and status == "proposed")):
                title = row["title"].strip()
                if title:
                    external.append((video_id, title, label))
            elif source not in {"personal", "external"}:
                raise ValueError(f"Unknown training source: {source}")
    return personal, external


def classify(source: Path, label_file: Path | None, output: Path, *, days: int = 90,
             external_labels: Path | None = None, training_data: Path | None = None,
             allow_proposed_external: bool = False, model_output: Path | None = None) -> None:
    # Build the prediction set from recent personal watch events only.
    model_output = model_output or output.with_suffix(".joblib") # 이전에 저장된 모델이 없으면 output 경로에 .joblib 확장자를 붙여서 저장
    if output.resolve() == model_output.resolve(): # resolve는 상대경로를 절대경로로 바꿔주는 함수임.
        raise ValueError("Prediction CSV and model file must have different paths")
    if output.exists() or model_output.exists():
        raise FileExistsError("Prediction CSV or model file already exists")
    if days < 1:
        raise ValueError("days must be positive")
    events = read_watch_events(source)
    if not events:
        raise ValueError("No watch events found")
    from datetime import timedelta
    # 영상 분석의 기간을 정함
    cutoff = max(event.watched_at for event in events) - timedelta(days=days)
    recent = [event for event in events if event.watched_at >= cutoff]
    titles = {event.video_id: event.title for event in recent}
    counts = Counter(event.video_id for event in recent)
    # Read labels from either a combined training file or separate CSV files.
    if training_data:
        if label_file or external_labels:
            raise ValueError("Use either --training-data or separate label files")
        labels, external = read_training_data(training_data, allow_proposed=allow_proposed_external)
    else:
        if label_file is None:
            raise ValueError("A label CSV or --training-data is required")
        labels = read_labels(label_file)
        external = read_external_labels(external_labels, allow_proposed=allow_proposed_external) if external_labels else []
    known = sorted(set(titles) & set(labels)) # 개인 레이블이 있는 영상만 남김
    external = [row for row in external if row[0] not in labels] # 외부 데이터 중에서 개인 레이블에 없는 것만 남김
    classes = Counter([labels[video_id] for video_id in known] + [row[2] for row in external]) # 개인레이블 + 외부 합쳐서 카테고리별 개수를 셈

    if len(classes) < 2:
        raise ValueError("At least two labeled categories are needed")
    # Embed personal titles and external training titles in the same vector space.
    ids = sorted(titles)
    # 텍스트 임베딩 모델을 사용하여 개인 영상 제목과 외부 학습 제목을 벡터로 변환함
    all_vectors = embed_titles([titles[video_id] for video_id in ids] + [row[1] for row in external])
    vectors = all_vectors[:len(ids)]
    external_vectors = all_vectors[len(ids):]
    index = {video_id: i for i, video_id in enumerate(ids)}
    x = np.asarray([vectors[index[video_id]] for video_id in known]) # 벡터값들을 저장
    y = [labels[video_id] for video_id in known] # 벡터값에 해당하는 레이블 저장
    personal_classes = Counter(y)

    # Evaluate on held-out personal labels when there are enough examples.
    # 가지고 있는 데이터를 테스트 데이터 구분하여 사용. 테스트용 따로 학습용 따로 만듬
    if len(personal_classes) >= 2 and min(personal_classes.values()) >= 2 and len(known) >= 10:
        test_count = max(len(personal_classes), round(len(known) * 0.2))
        if len(known) - test_count >= len(personal_classes):
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_count, random_state=42, stratify=y # 파일 분할을 할때 레이블별로 비율을 맞춰서 분할함. 시드를 고정시켜 같은 결과를 얻을 수 있게 함
            )
            if external:
                x_train = np.concatenate((x_train, external_vectors))
                y_train = list(y_train) + [row[2] for row in external]
            evaluation_model = LogisticRegression(max_iter=1000, class_weight="balanced")
            evaluation_model.fit(x_train, y_train)
            print(classification_report(y_test, evaluation_model.predict(x_test), zero_division=0))
    if external:
        x = np.concatenate((x, external_vectors))
        y = list(y) + [row[2] for row in external]
    # Fit the final classifier on all allowed labels, then score personal videos.
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x, y)
    probabilities = model.predict_proba(vectors)
    # Save predictions with review flags; external videos never become profile rows.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("video_id", "title", "watch_count", "category_label", "predicted_label", "confidence", "review_status"))
        for video_id, scores in zip(ids, probabilities):
            best = int(np.argmax(scores))
            confidence = float(scores[best])
            writer.writerow((video_id, titles[video_id], counts[video_id], labels.get(video_id, ""),
                             model.classes_[best], f"{confidence:.4f}",
                             "labeled" if video_id in labels else "suggested" if confidence >= 0.8 else "review_needed"))
    # Save the fitted classifier and the embedding settings needed to reuse it.
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("xb") as file:
        joblib.dump({
            "classifier": model,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": int(vectors.shape[1]),
            "scikit_learn_version": sklearn.__version__,
            "days": days,
        }, file)
    print(f"개인 학습 레이블: {len(known)}개 / 외부 학습 레이블: {len(external)}개 / 카테고리: {len(classes)}개")
    print(f"예측 대상: 개인 시청 영상 {len(ids)}개")
    print(f"결과: {output}")
    print(f"학습 모델: {model_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify YouTube titles using reviewed labels")
    parser.add_argument("source", type=Path)
    parser.add_argument("labels", nargs="?", type=Path)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--external-labels", type=Path)
    parser.add_argument("--training-data", type=Path,
                        help="Combined training CSV with source and label_status columns")
    parser.add_argument("--allow-proposed-external", action="store_true",
                        help="Include unverified search-based external labels in training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path,
                        help="Private model path (default: prediction CSV path with .joblib extension)")
    args = parser.parse_args()
    try:
        classify(args.source, args.labels, args.output, days=args.days,
                 external_labels=args.external_labels,
                 training_data=args.training_data,
                 allow_proposed_external=args.allow_proposed_external,
                 model_output=args.model_output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
