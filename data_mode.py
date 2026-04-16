from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from core import (
    EPSILON,
    Matrix,
    calculate_two_scores,
    decide_label,
    normalize_label,
    validate_matrix,
)


@dataclass
class AnalysisResult:
    """data.json의 패턴 한 건을 분석한 결과를 담는 상자다.

    점수, 예측 라벨, PASS/FAIL 여부, 실패 이유를 한 곳에 모아 두면
    출력 함수가 복잡한 계산 과정을 몰라도 결과를 깔끔하게 보여 줄 수 있다.
    """

    case_id: str
    expected: str | None
    prediction: str
    score_cross: float | None
    score_x: float | None
    passed: bool
    reason: str = ""


def extract_size_from_key(case_id: str) -> int | None:
    """'size_13_1' 같은 케이스 이름에서 패턴 크기 13을 꺼낸다.

    이름 형식이 맞지 않으면 None을 반환해서 해당 케이스를 안전하게 FAIL 처리하게 한다.
    """
    match = re.fullmatch(r"size_(\d+)_(\d+)", case_id)
    if match is None:
        return None
    return int(match.group(1))


def pattern_sort_key(case_id: str) -> tuple[int, int, str]:
    """패턴 케이스를 크기와 번호 순서로 보기 좋게 정렬하기 위한 기준을 만든다.

    예를 들어 size_5_1, size_5_2, size_13_1 순서로 출력되게 한다.
    형식이 이상한 키는 맨 뒤쪽으로 보내서 정상 케이스를 먼저 볼 수 있게 한다.
    """
    match = re.fullmatch(r"size_(\d+)_(\d+)", case_id)
    if match is None:
        return (10**9, 10**9, case_id)
    return (int(match.group(1)), int(match.group(2)), case_id)


def sort_filter_key(filter_key: str) -> tuple[int, str]:
    """필터 묶음을 size_5, size_13, size_25처럼 크기 순서로 정렬하기 위한 기준을 만든다."""
    match = re.fullmatch(r"size_(\d+)", filter_key)
    if match is None:
        return (10**9, filter_key)
    return (int(match.group(1)), filter_key)


def load_json_payload(data_file: Path) -> dict[str, object]:
    """data.json 파일을 읽고 최상위 구조가 JSON 객체인지 확인한다.

    프로그램은 filters와 patterns라는 이름으로 데이터를 찾아야 하므로,
    최상위가 배열이나 문자열이면 이후 처리를 진행할 수 없다.
    """
    with data_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("data.json의 최상위 구조는 객체(object)여야 합니다.")

    return payload


def load_filters(payload: dict[str, object]) -> tuple[dict[int, dict[str, Matrix]], list[str]]:
    """data.json의 filters 영역을 읽어 크기별 Cross/X 필터로 정리한다.

    필터 키의 size_N 규칙, cross/x 라벨 정규화, 행렬 크기 검증을 모두 수행한다.
    실패한 필터 묶음은 메시지로 남기고, 성공한 필터만 분석에 사용할 수 있게 돌려준다.
    """
    raw_filters = payload.get("filters")
    if not isinstance(raw_filters, dict):
        raise ValueError("data.json에 filters 객체가 없습니다.")

    messages: list[str] = []
    filters_by_size: dict[int, dict[str, Matrix]] = {}

    for filter_key in sorted(raw_filters.keys(), key=sort_filter_key):
        raw_bundle = raw_filters[filter_key]

        if not isinstance(raw_bundle, dict):
            messages.append(f"✗ {filter_key} 필터 로드 실패: Cross/X 필터 묶음이 아닙니다.")
            continue

        size_match = re.fullmatch(r"size_(\d+)", filter_key)
        if size_match is None:
            messages.append(f"✗ {filter_key} 필터 로드 실패: size_N 형식이 아닙니다.")
            continue

        size = int(size_match.group(1))
        bundle: dict[str, Matrix] = {}
        errors: list[str] = []

        for raw_label, raw_matrix in raw_bundle.items():
            label = normalize_label(raw_label)
            if label is None:
                errors.append(f"알 수 없는 필터 라벨({raw_label})")
                continue

            try:
                bundle[label] = validate_matrix(raw_matrix, expected_size=size, name=f"{filter_key}.{raw_label}")
            except ValueError as error:
                errors.append(str(error))

        if "Cross" not in bundle or "X" not in bundle:
            errors.append("Cross/X 필터가 모두 필요합니다.")

        if errors:
            messages.append(f"✗ {filter_key} 필터 로드 실패: {'; '.join(errors)}")
            continue

        filters_by_size[size] = bundle
        messages.append(f"✓ {filter_key} 필터 로드 완료 (Cross, X)")

    return filters_by_size, messages


def analyze_pattern_case(case_id: str, case_payload: object, filters_by_size: dict[int, dict[str, Matrix]]) -> AnalysisResult:
    """data.json의 패턴 한 건을 검증하고 Cross/X 판정을 수행한다.

    케이스 키에서 크기를 찾고, expected 라벨을 표준화하고, 입력 행렬 크기를 확인한다.
    문제가 있으면 프로그램을 멈추지 않고 AnalysisResult에 실패 이유를 담아 반환한다.
    """
    if not isinstance(case_payload, dict):
        return AnalysisResult(case_id, None, "ERROR", None, None, False, "케이스 구조가 객체가 아닙니다.")

    size = extract_size_from_key(case_id)
    if size is None:
        return AnalysisResult(case_id, None, "ERROR", None, None, False, "케이스 키가 size_N_idx 형식이 아닙니다.")

    if size not in filters_by_size:
        return AnalysisResult(case_id, None, "ERROR", None, None, False, f"size_{size} 필터를 찾을 수 없습니다.")

    expected = normalize_label(case_payload.get("expected"))
    if expected is None:
        raw_expected = case_payload.get("expected")
        return AnalysisResult(case_id, None, "ERROR", None, None, False, f"expected 라벨을 정규화할 수 없습니다: {raw_expected}")

    try:
        pattern = validate_matrix(case_payload.get("input"), expected_size=size, name=f"{case_id}.input")
    except ValueError as error:
        return AnalysisResult(case_id, expected, "ERROR", None, None, False, str(error))

    cross_filter = filters_by_size[size]["Cross"]
    x_filter = filters_by_size[size]["X"]
    score_cross, score_x = calculate_two_scores(pattern, cross_filter, x_filter)
    prediction = decide_label(score_cross, score_x)

    if prediction == expected:
        return AnalysisResult(case_id, expected, prediction, score_cross, score_x, True)

    if prediction == "UNDECIDED":
        reason = f"동점 규칙 적용 (|Cross-X| < {EPSILON})"
    else:
        reason = f"expected={expected}, prediction={prediction}"

    return AnalysisResult(case_id, expected, prediction, score_cross, score_x, False, reason)
