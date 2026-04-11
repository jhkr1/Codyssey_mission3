from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

EPSILON = 1e-9
REPEAT_COUNT = 10
DATA_FILE = Path(__file__).with_name("data.json")

Matrix = list[list[float]]


@dataclass
class AnalysisResult:
    case_id: str
    expected: str | None
    prediction: str
    score_cross: float | None
    score_x: float | None
    passed: bool
    reason: str = ""


def generate_cross_pattern(size: int) -> Matrix:
    if size < 1 or size % 2 == 0:
        raise ValueError("패턴 생성기는 1 이상의 홀수 크기만 지원합니다.")

    middle = size // 2
    return [
        [1.0 if row == middle or col == middle else 0.0 for col in range(size)]
        for row in range(size)
    ]


def generate_x_pattern(size: int) -> Matrix:
    if size < 1 or size % 2 == 0:
        raise ValueError("패턴 생성기는 1 이상의 홀수 크기만 지원합니다.")

    return [
        [1.0 if row == col or row + col == size - 1 else 0.0 for col in range(size)]
        for row in range(size)
    ]


def normalize_label(raw_label: object) -> str | None:
    if not isinstance(raw_label, str):
        return None

    normalized = raw_label.strip().lower()
    mapping = {
        "+": "Cross",
        "cross": "Cross",
        "x": "X",
    }
    return mapping.get(normalized)


def format_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.16f}"


def extract_size_from_key(pattern_key: str) -> int | None:
    match = re.fullmatch(r"size_(\d+)_(\d+)", pattern_key)
    if match is None:
        return None
    return int(match.group(1))


def extract_pattern_sort_key(pattern_key: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"size_(\d+)_(\d+)", pattern_key)
    if match is None:
        return (10**9, 10**9, pattern_key)
    return (int(match.group(1)), int(match.group(2)), pattern_key)


def coerce_matrix(raw_matrix: object, *, expected_size: int | None = None, matrix_name: str = "matrix") -> Matrix:
    if not isinstance(raw_matrix, list) or not raw_matrix:
        raise ValueError(f"{matrix_name}는 비어 있지 않은 2차원 배열이어야 합니다.")

    row_count = len(raw_matrix)
    if expected_size is not None and row_count != expected_size:
        raise ValueError(f"{matrix_name}의 행 수가 {expected_size}가 아닙니다.")

    matrix: Matrix = []
    for row_index, raw_row in enumerate(raw_matrix, start=1):
        if not isinstance(raw_row, list):
            raise ValueError(f"{matrix_name}의 {row_index}행이 배열 형식이 아닙니다.")

        if len(raw_row) != row_count:
            raise ValueError(f"{matrix_name}는 정사각형 2차원 배열이어야 합니다.")

        if expected_size is not None and len(raw_row) != expected_size:
            raise ValueError(f"{matrix_name}의 열 수가 {expected_size}가 아닙니다.")

        row: list[float] = []
        for col_index, value in enumerate(raw_row, start=1):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{matrix_name}의 {row_index}행 {col_index}열 값은 숫자여야 합니다."
                )
            row.append(float(value))
        matrix.append(row)

    return matrix


def mac(pattern: Matrix, matrix_filter: Matrix) -> float:
    if len(pattern) != len(matrix_filter):
        raise ValueError("패턴과 필터의 크기가 다릅니다.")

    total = 0.0
    size = len(pattern)
    for row in range(size):
        if len(pattern[row]) != len(matrix_filter[row]):
            raise ValueError("패턴과 필터의 열 수가 다릅니다.")

        for col in range(size):
            total += pattern[row][col] * matrix_filter[row][col]

    return total


def measure_average_ms(operation: Callable[[], object], repeat: int = REPEAT_COUNT) -> float:
    durations_ms: list[float] = []
    for _ in range(repeat):
        started_at = time.perf_counter()
        operation()
        ended_at = time.perf_counter()
        durations_ms.append((ended_at - started_at) * 1000)

    return sum(durations_ms) / len(durations_ms)


def decide_label(score_cross: float, score_x: float, epsilon: float = EPSILON) -> str:
    if abs(score_cross - score_x) < epsilon:
        return "UNDECIDED"
    if score_cross > score_x:
        return "Cross"
    return "X"


def read_numeric_row(size: int, row_number: int) -> list[float]:
    raw_line = input(f"{row_number}행> ").strip()
    tokens = raw_line.split()

    if len(tokens) != size:
        raise ValueError(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")

    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError as error:
            raise ValueError(
                f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
            ) from error

    return values


def read_console_matrix(title: str, size: int) -> Matrix:
    print(title)
    rows: Matrix = []

    while len(rows) < size:
        try:
            rows.append(read_numeric_row(size, len(rows) + 1))
        except ValueError as error:
            print(error)

    return rows


def benchmark_single_mac(size: int, repeat: int = REPEAT_COUNT) -> tuple[float, int]:
    pattern = generate_cross_pattern(size)
    matrix_filter = generate_cross_pattern(size)
    average_ms = measure_average_ms(lambda: mac(pattern, matrix_filter), repeat)
    return average_ms, size * size


def load_json_payload(data_file: Path) -> dict[str, object]:
    with data_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("data.json의 최상위 구조는 객체(object)여야 합니다.")

    return payload


def load_filters(payload: dict[str, object]) -> tuple[dict[int, dict[str, Matrix]], list[str]]:
    raw_filters = payload.get("filters")
    if not isinstance(raw_filters, dict):
        raise ValueError("data.json에 filters 객체가 없습니다.")

    loaded: dict[int, dict[str, Matrix]] = {}
    messages: list[str] = []
    filter_key_pattern = re.compile(r"size_(\d+)")

    def filter_sort_key(value: str) -> tuple[int, str]:
        match = filter_key_pattern.fullmatch(value)
        if match is None:
            return (10**9, value)
        return (int(match.group(1)), value)

    for filter_key in sorted(raw_filters.keys(), key=filter_sort_key):
        bundle = raw_filters[filter_key]
        if not isinstance(bundle, dict):
            messages.append(f"✗ {filter_key} 필터 로드 실패: Cross/X 필터 묶음이 아닙니다.")
            continue

        match = filter_key_pattern.fullmatch(filter_key)
        if match is None:
            messages.append(f"✗ {filter_key} 필터 로드 실패: size_N 형식이 아닙니다.")
            continue

        size = int(match.group(1))
        normalized_bundle: dict[str, Matrix] = {}
        bundle_errors: list[str] = []

        for raw_label, raw_matrix in bundle.items():
            label = normalize_label(raw_label)
            if label is None:
                bundle_errors.append(f"알 수 없는 필터 라벨({raw_label})")
                continue

            try:
                normalized_bundle[label] = coerce_matrix(
                    raw_matrix,
                    expected_size=size,
                    matrix_name=f"{filter_key}.{raw_label}",
                )
            except ValueError as error:
                bundle_errors.append(str(error))

        if "Cross" not in normalized_bundle or "X" not in normalized_bundle:
            bundle_errors.append("Cross/X 필터가 모두 필요합니다.")

        if bundle_errors:
            joined_errors = "; ".join(bundle_errors)
            messages.append(f"✗ {filter_key} 필터 로드 실패: {joined_errors}")
            continue

        loaded[size] = normalized_bundle
        messages.append(f"✓ {filter_key} 필터 로드 완료 (Cross, X)")

    return loaded, messages


def analyze_pattern_case(
    case_id: str,
    case_payload: object,
    filters_by_size: dict[int, dict[str, Matrix]],
) -> AnalysisResult:
    if not isinstance(case_payload, dict):
        return AnalysisResult(case_id, None, "ERROR", None, None, False, "케이스 구조가 객체가 아닙니다.")

    size = extract_size_from_key(case_id)
    if size is None:
        return AnalysisResult(case_id, None, "ERROR", None, None, False, "케이스 키가 size_N_idx 형식이 아닙니다.")

    if size not in filters_by_size:
        return AnalysisResult(case_id, None, "ERROR", None, None, False, f"size_{size} 필터를 찾을 수 없습니다.")

    raw_expected = case_payload.get("expected")
    expected = normalize_label(raw_expected)
    if expected is None:
        return AnalysisResult(case_id, None, "ERROR", None, None, False, f"expected 라벨을 정규화할 수 없습니다: {raw_expected}")

    try:
        pattern = coerce_matrix(case_payload.get("input"), expected_size=size, matrix_name=f"{case_id}.input")
    except ValueError as error:
        return AnalysisResult(case_id, expected, "ERROR", None, None, False, str(error))

    score_cross = mac(pattern, filters_by_size[size]["Cross"])
    score_x = mac(pattern, filters_by_size[size]["X"])
    prediction = decide_label(score_cross, score_x)

    if prediction == expected:
        return AnalysisResult(case_id, expected, prediction, score_cross, score_x, True)

    if prediction == "UNDECIDED":
        reason = f"동점 규칙 적용 (|Cross-X| < {EPSILON})"
    else:
        reason = f"expected={expected}, prediction={prediction}"

    return AnalysisResult(case_id, expected, prediction, score_cross, score_x, False, reason)


def print_performance_table(sizes: list[int], section_number: int = 3) -> None:
    print("#---------------------------------------")
    print(f"# [{section_number}] 성능 분석 (평균/{REPEAT_COUNT}회, 단일 MAC 기준)")
    print("#---------------------------------------")
    print(f"{'크기':<10}{'평균 시간(ms)':<18}{'연산 횟수(N²)':<15}")
    print("-" * 43)
    for size in sizes:
        average_ms, operations = benchmark_single_mac(size)
        size_label = f"{size}x{size}"
        print(f"{size_label:<10}{average_ms:<18.6f}{operations:<15}")
    print()


def run_user_input_mode() -> None:
    print()
    print("#---------------------------------------")
    print("# [1] 필터 입력")
    print("#---------------------------------------")
    filter_a = read_console_matrix("필터 A (3줄 입력, 공백 구분)", 3)
    print()
    filter_b = read_console_matrix("필터 B (3줄 입력, 공백 구분)", 3)

    print()
    print("#---------------------------------------")
    print("# [2] 패턴 입력")
    print("#---------------------------------------")
    pattern = read_console_matrix("패턴 (3줄 입력, 공백 구분)", 3)

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    average_ms = measure_average_ms(lambda: (mac(pattern, filter_a), mac(pattern, filter_b)))
    decision = "판정 불가" if abs(score_a - score_b) < EPSILON else ("A" if score_a > score_b else "B")

    print()
    print("#---------------------------------------")
    print("# [3] MAC 결과")
    print("#---------------------------------------")
    print(f"A 점수: {format_score(score_a)}")
    print(f"B 점수: {format_score(score_b)}")
    print(f"연산 시간(평균/{REPEAT_COUNT}회, 필터 2개 기준): {average_ms:.6f} ms")
    if decision == "판정 불가":
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    else:
        print(f"판정: {decision}")
    print()

    print_performance_table([3], section_number=4)


def run_json_mode() -> None:
    print()
    print("#---------------------------------------")
    print("# [1] 필터 로드")
    print("#---------------------------------------")

    try:
        payload = load_json_payload(DATA_FILE)
        filters_by_size, filter_messages = load_filters(payload)
    except FileNotFoundError:
        print("data.json 파일을 찾을 수 없습니다.")
        return
    except json.JSONDecodeError as error:
        print(f"data.json 파싱 실패: {error}")
        return
    except ValueError as error:
        print(f"data.json 구조 오류: {error}")
        return

    for message in filter_messages:
        print(message)

    raw_patterns = payload.get("patterns")
    if not isinstance(raw_patterns, dict):
        print("data.json 구조 오류: patterns 객체가 없습니다.")
        return

    print()
    print("#---------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#---------------------------------------")

    results: list[AnalysisResult] = []
    for case_id in sorted(raw_patterns.keys(), key=extract_pattern_sort_key):
        result = analyze_pattern_case(case_id, raw_patterns[case_id], filters_by_size)
        results.append(result)

        print(f"--- {case_id} ---")
        print(f"Cross 점수: {format_score(result.score_cross)}")
        print(f"X 점수: {format_score(result.score_x)}")
        if result.expected is None:
            expected_label = "N/A"
        else:
            expected_label = result.expected
        outcome = "PASS" if result.passed else "FAIL"

        if result.reason:
            print(f"판정: {result.prediction} | expected: {expected_label} | {outcome} ({result.reason})")
        else:
            print(f"판정: {result.prediction} | expected: {expected_label} | {outcome}")
        print()

    print_performance_table([3, 5, 13, 25])

    total_count = len(results)
    passed_count = sum(1 for result in results if result.passed)
    failed_results = [result for result in results if not result.passed]

    print("#---------------------------------------")
    print("# [4] 결과 요약")
    print("#---------------------------------------")
    print(f"총 테스트: {total_count}개")
    print(f"통과: {passed_count}개")
    print(f"실패: {len(failed_results)}개")

    if failed_results:
        print()
        print("실패 케이스:")
        for result in failed_results:
            print(f"- {result.case_id}: {result.reason}")


def choose_mode() -> str:
    while True:
        print()
        print("[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        choice = input("선택: ").strip()
        if choice in {"1", "2"}:
            return choice
        print("메뉴 입력 오류: 1 또는 2를 입력하세요.")


def main() -> None:
    print("=== Mini NPU Simulator ===")
    selected_mode = choose_mode()

    if selected_mode == "1":
        run_user_input_mode()
    else:
        run_json_mode()


if __name__ == "__main__":
    main()
