import json
from pathlib import Path

from core import (
    EPSILON,
    REPEAT_COUNT,
    average_two_filter_ms,
    benchmark_mac,
    calculate_mac,
    format_score,
    generate_cross_pattern,
    generate_pattern,
    generate_x_pattern,
)
from data_mode import analyze_pattern_case, load_filters, load_json_payload, pattern_sort_key

DATA_FILE = Path(__file__).with_name("data.json")


def print_matrix(title, matrix):
    """2차원 배열을 사람이 읽기 쉬운 표 모양으로 출력한다."""
    print(title)
    for row in matrix:
        print(" ".join(str(int(value)) if value.is_integer() else str(value) for value in row))


def read_numeric_row(size, row_number):
    """콘솔에서 숫자 한 줄을 입력받아 float 리스트로 바꾼다.

    사용자가 숫자가 아닌 값을 넣거나, 필요한 개수보다 적거나 많이 입력하면
    ValueError를 발생시켜 다시 입력하도록 안내할 수 있게 한다.
    """
    raw_line = input(f"{row_number}행> ").strip()
    values = raw_line.split()

    if len(values) != size:
        raise ValueError(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")

    row = []
    for value in values:
        try:
            row.append(float(value))
        except ValueError:
            raise ValueError(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")

    return row


def read_matrix_from_console(title, size):
    """사용자에게 size줄을 입력받아 size x size 행렬을 만든다.

    한 줄 입력이 잘못되면 프로그램을 종료하지 않고 오류 메시지를 보여 준 뒤
    같은 줄을 다시 입력받는다.
    """
    print(title)
    matrix = []

    while len(matrix) < size:
        try:
            matrix.append(read_numeric_row(size, len(matrix) + 1))
        except ValueError as error:
            print(error)

    return matrix


def choose_from_menu(title, options):
    """메뉴를 출력하고 사용자가 올바른 번호를 고를 때까지 반복한다."""
    while True:
        print(title)
        for key, text in options.items():
            print(f"{key}. {text}")
        choice = input("선택: ").strip()
        if choice in options:
            return choice
        print(f"메뉴 입력 오류: {', '.join(options.keys())} 중 하나를 입력하세요.")
        print()


def choose_generated_label():
    """자동 생성 예제에서 만들 패턴이 Cross인지 X인지 선택받는다."""
    choice = choose_from_menu(
        "[자동 생성 패턴 선택]",
        {
            "1": "Cross 패턴 생성",
            "2": "X 패턴 생성",
        },
    )
    return "Cross" if choice == "1" else "X"


def get_user_mode_matrices():
    """사용자 입력 모드에서 필터 A, 필터 B, 패턴을 준비한다.

    직접 입력을 고르면 콘솔에서 3x3 행렬을 받는다.
    자동 생성 예제를 고르면 Cross 필터, X 필터, 선택한 패턴을 코드가 대신 만든다.
    """
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


def print_performance_table(sizes, section_number):
    """여러 크기에 대해 MAC 평균 실행 시간과 연산 횟수를 표로 출력한다.

    2D 방식과 1D 방식의 시간을 함께 보여 주어 보너스 최적화 효과도 확인할 수 있다.
    """
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


def print_user_mode_result(score_a, score_b, average_ms_value):
    """사용자 입력 모드의 MAC 점수, 평균 시간, 최종 판정을 출력한다."""
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


def run_user_input_mode():
    """사용자 입력 모드 전체 흐름을 실행한다.

    필터와 패턴을 준비하고, 두 필터의 MAC 점수를 계산한 뒤,
    결과와 3x3 성능 분석 표를 차례대로 보여 준다.
    """
    print()
    filter_a, filter_b, pattern = get_user_mode_matrices()

    score_a = calculate_mac(pattern, filter_a)
    score_b = calculate_mac(pattern, filter_b)
    average_time = average_two_filter_ms(pattern, filter_a, filter_b)

    print_user_mode_result(score_a, score_b, average_time)
    print_performance_table([3], section_number=5)


def print_case_result(result):
    """data.json 패턴 한 건의 분석 결과를 콘솔에 출력한다."""
    expected_label = result["expected"] if result["expected"] is not None else "N/A"
    outcome = "PASS" if result["passed"] else "FAIL"

    print(f"--- {result['case_id']} ---")
    print(f"Cross 점수: {format_score(result['score_cross'])}")
    print(f"X 점수: {format_score(result['score_x'])}")

    if result["reason"]:
        print(f"판정: {result['prediction']} | expected: {expected_label} | {outcome} ({result['reason']})")
    else:
        print(f"판정: {result['prediction']} | expected: {expected_label} | {outcome}")
    print()


def print_summary(results):
    """data.json 분석 전체 결과의 총 테스트, 통과, 실패 수를 요약한다."""
    total_count = len(results)
    passed_count = sum(1 for result in results if result["passed"])
    failed_results = [result for result in results if not result["passed"]]

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
            print(f"- {result['case_id']}: {result['reason']}")


def run_json_mode():
    """data.json 분석 모드 전체 흐름을 실행한다.

    파일 로드, 필터 검증, 패턴별 MAC 판정, 성능 분석, 결과 요약을 순서대로 진행한다.
    JSON 구조가 잘못된 경우에는 오류 메시지를 출력하고 안전하게 종료한다.
    """
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

    results = []
    for case_id in sorted(raw_patterns.keys(), key=pattern_sort_key):
        result = analyze_pattern_case(case_id, raw_patterns[case_id], filters_by_size)
        results.append(result)
        print_case_result(result)

    print_performance_table([3, 5, 13, 25], section_number=3)
    print_summary(results)


def choose_mode():
    """프로그램 시작 시 사용자 입력 모드와 data.json 분석 모드 중 하나를 선택받는다."""
    return choose_from_menu(
        "[모드 선택]",
        {
            "1": "사용자 입력 (3x3)",
            "2": "data.json 분석",
        },
    )


def main():
    """Mini NPU Simulator의 시작점이다."""
    print("=== Mini NPU Simulator ===")
    selected_mode = choose_mode()

    if selected_mode == "1":
        run_user_input_mode()
    else:
        run_json_mode()


if __name__ == "__main__":
    main()
