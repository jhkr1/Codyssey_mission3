from __future__ import annotations

import json
from pathlib import Path

from core import (
    EPSILON,
    REPEAT_COUNT,
    Matrix,
    average_ms,
    benchmark_mac,
    calculate_mac,
    format_score,
    generate_cross_pattern,
    generate_pattern,
    generate_x_pattern,
)
from data_mode import AnalysisResult, analyze_pattern_case, load_filters, load_json_payload, pattern_sort_key

DATA_FILE = Path(__file__).with_name("data.json")


def print_matrix(title: str, matrix: Matrix) -> None:
    print(title)
    for row in matrix:
        print(" ".join(str(int(value)) if value.is_integer() else str(value) for value in row))


def read_numeric_row(size: int, row_number: int) -> list[float]:
    raw_line = input(f"{row_number}행> ").strip()
    values = raw_line.split()

    if len(values) != size:
        raise ValueError(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")

    row: list[float] = []
    for value in values:
        try:
            row.append(float(value))
        except ValueError as error:
            raise ValueError(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.") from error

    return row


def read_matrix_from_console(title: str, size: int) -> Matrix:
    print(title)
    matrix: Matrix = []

    while len(matrix) < size:
        try:
            matrix.append(read_numeric_row(size, len(matrix) + 1))
        except ValueError as error:
            print(error)

    return matrix


def choose_from_menu(title: str, options: dict[str, str]) -> str:
    while True:
        print(title)
        for key, text in options.items():
            print(f"{key}. {text}")
        choice = input("선택: ").strip()
        if choice in options:
            return choice
        print(f"메뉴 입력 오류: {', '.join(options.keys())} 중 하나를 입력하세요.")
        print()


def choose_generated_label() -> str:
    choice = choose_from_menu(
        "[자동 생성 패턴 선택]",
        {
            "1": "Cross 패턴 생성",
            "2": "X 패턴 생성",
        },
    )
    return "Cross" if choice == "1" else "X"


def get_user_mode_matrices() -> tuple[Matrix, Matrix, Matrix]:
    print()
    mode = choose_from_menu(
        "[1] 입력 방식 선택",
        {
            "1": "직접 입력",
            "2": "자동 생성 예제 사용",
        },
    )

    if mode == "1":
        print()
        print("#---------------------------------------")
        print("# [2] 필터 입력")
        print("#---------------------------------------")
        filter_a = read_matrix_from_console("필터 A (3줄 입력, 공백 구분)", 3)
        print()
        filter_b = read_matrix_from_console("필터 B (3줄 입력, 공백 구분)", 3)

        print()
        print("#---------------------------------------")
        print("# [3] 패턴 입력")
        print("#---------------------------------------")
        pattern = read_matrix_from_console("패턴 (3줄 입력, 공백 구분)", 3)
        return filter_a, filter_b, pattern

    pattern_label = choose_generated_label()
    filter_a = generate_cross_pattern(3)
    filter_b = generate_x_pattern(3)
    pattern = generate_pattern(pattern_label, 3)

    print()
    print("#---------------------------------------")
    print("# [2] 자동 생성 결과")
    print("#---------------------------------------")
    print_matrix("필터 A = Cross", filter_a)
    print()
    print_matrix("필터 B = X", filter_b)
    print()
    print_matrix(f"패턴 = {pattern_label}", pattern)

    return filter_a, filter_b, pattern


def print_performance_table(sizes: list[int], section_number: int) -> None:
    print("#---------------------------------------")
    print(f"# [{section_number}] 성능 분석 (평균/{REPEAT_COUNT}회, 2D vs 1D 비교)")
    print("#---------------------------------------")
    print(f"{'크기':<10}{'2D(ms)':<14}{'1D(ms)':<14}{'개선율':<12}{'연산 횟수(N²)':<15}")
    print("-" * 65)

    for size in sizes:
        average_2d, average_1d, operations, improvement = benchmark_mac(size)
        size_label = f"{size}x{size}"
        improvement_text = f"{improvement:.2f}%"
        print(f"{size_label:<10}{average_2d:<14.6f}{average_1d:<14.6f}{improvement_text:<12}{operations:<15}")

    print()


def print_user_mode_result(score_a: float, score_b: float, average_ms_value: float) -> None:
    print()
    print("#---------------------------------------")
    print("# [4] MAC 결과")
    print("#---------------------------------------")
    print(f"A 점수: {format_score(score_a)}")
    print(f"B 점수: {format_score(score_b)}")
    print(f"연산 시간(평균/{REPEAT_COUNT}회, 필터 2개 기준): {average_ms_value:.6f} ms")

    if abs(score_a - score_b) < EPSILON:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    elif score_a > score_b:
        print("판정: A")
    else:
        print("판정: B")

    print()


def run_user_input_mode() -> None:
    print()
    filter_a, filter_b, pattern = get_user_mode_matrices()

    score_a = calculate_mac(pattern, filter_a)
    score_b = calculate_mac(pattern, filter_b)
    average_time = average_ms(lambda: (calculate_mac(pattern, filter_a), calculate_mac(pattern, filter_b)))

    print_user_mode_result(score_a, score_b, average_time)
    print_performance_table([3], section_number=5)


def print_case_result(result: AnalysisResult) -> None:
    expected_label = result.expected if result.expected is not None else "N/A"
    outcome = "PASS" if result.passed else "FAIL"

    print(f"--- {result.case_id} ---")
    print(f"Cross 점수: {format_score(result.score_cross)}")
    print(f"X 점수: {format_score(result.score_x)}")

    if result.reason:
        print(f"판정: {result.prediction} | expected: {expected_label} | {outcome} ({result.reason})")
    else:
        print(f"판정: {result.prediction} | expected: {expected_label} | {outcome}")
    print()


def print_summary(results: list[AnalysisResult]) -> None:
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


def run_json_mode() -> None:
    print()
    print("#---------------------------------------")
    print("# [1] 필터 로드")
    print("#---------------------------------------")

    try:
        payload = load_json_payload(DATA_FILE)
        filters_by_size, messages = load_filters(payload)
    except FileNotFoundError:
        print("data.json 파일을 찾을 수 없습니다.")
        return
    except json.JSONDecodeError as error:
        print(f"data.json 파싱 실패: {error}")
        return
    except ValueError as error:
        print(f"data.json 구조 오류: {error}")
        return

    for message in messages:
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
    for case_id in sorted(raw_patterns.keys(), key=pattern_sort_key):
        result = analyze_pattern_case(case_id, raw_patterns[case_id], filters_by_size)
        results.append(result)
        print_case_result(result)

    print_performance_table([3, 5, 13, 25], section_number=3)
    print_summary(results)


def choose_mode() -> str:
    return choose_from_menu(
        "[모드 선택]",
        {
            "1": "사용자 입력 (3x3)",
            "2": "data.json 분석",
        },
    )


def main() -> None:
    print("=== Mini NPU Simulator ===")
    selected_mode = choose_mode()

    if selected_mode == "1":
        run_user_input_mode()
    else:
        run_json_mode()


if __name__ == "__main__":
    main()
