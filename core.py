from __future__ import annotations

import time
from typing import Callable

EPSILON = 1e-9
REPEAT_COUNT = 10

Matrix = list[list[float]]
FlatMatrix = list[float]


def normalize_label(raw_label: object) -> str | None:
    if not isinstance(raw_label, str):
        return None

    mapping = {
        "+": "Cross",
        "cross": "Cross",
        "x": "X",
    }
    return mapping.get(raw_label.strip().lower())


def format_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.16f}"


def validate_generator_size(size: int) -> None:
    if size < 1 or size % 2 == 0:
        raise ValueError("패턴 생성기는 1 이상의 홀수 크기만 지원합니다.")


def create_empty_matrix(size: int) -> Matrix:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def generate_cross_pattern(size: int) -> Matrix:
    validate_generator_size(size)

    result = create_empty_matrix(size)
    middle = size // 2

    for row in range(size):
        for col in range(size):
            if row == middle or col == middle:
                result[row][col] = 1.0

    return result


def generate_x_pattern(size: int) -> Matrix:
    validate_generator_size(size)

    result = create_empty_matrix(size)

    for row in range(size):
        for col in range(size):
            if row == col or row + col == size - 1:
                result[row][col] = 1.0

    return result


def generate_pattern(label: str, size: int) -> Matrix:
    if label == "Cross":
        return generate_cross_pattern(size)
    if label == "X":
        return generate_x_pattern(size)
    raise ValueError(f"알 수 없는 패턴 라벨입니다: {label}")


def flatten_matrix(matrix: Matrix) -> FlatMatrix:
    flat: FlatMatrix = []
    for row in matrix:
        for value in row:
            flat.append(value)
    return flat


def validate_matrix(raw_matrix: object, expected_size: int | None = None, name: str = "matrix") -> Matrix:
    if not isinstance(raw_matrix, list) or not raw_matrix:
        raise ValueError(f"{name}는 비어 있지 않은 2차원 배열이어야 합니다.")

    size = len(raw_matrix)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{name}의 행 수가 {expected_size}가 아닙니다.")

    matrix: Matrix = []
    for row_index, raw_row in enumerate(raw_matrix, start=1):
        if not isinstance(raw_row, list):
            raise ValueError(f"{name}의 {row_index}행이 배열 형식이 아닙니다.")

        if len(raw_row) != size:
            raise ValueError(f"{name}는 정사각형 2차원 배열이어야 합니다.")

        if expected_size is not None and len(raw_row) != expected_size:
            raise ValueError(f"{name}의 열 수가 {expected_size}가 아닙니다.")

        row: list[float] = []
        for col_index, value in enumerate(raw_row, start=1):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}의 {row_index}행 {col_index}열 값은 숫자여야 합니다.")
            row.append(float(value))
        matrix.append(row)

    return matrix


def calculate_mac(pattern: Matrix, matrix_filter: Matrix) -> float:
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


def calculate_mac_flat(pattern: FlatMatrix, matrix_filter: FlatMatrix) -> float:
    if len(pattern) != len(matrix_filter):
        raise ValueError("1차원 패턴과 필터의 길이가 다릅니다.")

    total = 0.0
    for index in range(len(pattern)):
        total += pattern[index] * matrix_filter[index]

    return total


def average_ms(operation: Callable[[], object], repeat: int = REPEAT_COUNT) -> float:
    durations: list[float] = []

    for _ in range(repeat):
        start = time.perf_counter()
        operation()
        end = time.perf_counter()
        durations.append((end - start) * 1000)

    return sum(durations) / len(durations)


def decide_label(score_cross: float, score_x: float, epsilon: float = EPSILON) -> str:
    if abs(score_cross - score_x) < epsilon:
        return "UNDECIDED"
    if score_cross > score_x:
        return "Cross"
    return "X"


def calculate_two_scores(pattern: Matrix, cross_filter: Matrix, x_filter: Matrix) -> tuple[float, float]:
    score_cross = calculate_mac(pattern, cross_filter)
    score_x = calculate_mac(pattern, x_filter)
    return score_cross, score_x


def benchmark_mac(size: int, repeat: int = REPEAT_COUNT) -> tuple[float, float, int, float]:
    pattern_2d = generate_cross_pattern(size)
    filter_2d = generate_cross_pattern(size)
    pattern_1d = flatten_matrix(pattern_2d)
    filter_1d = flatten_matrix(filter_2d)

    average_2d = average_ms(lambda: calculate_mac(pattern_2d, filter_2d), repeat)
    average_1d = average_ms(lambda: calculate_mac_flat(pattern_1d, filter_1d), repeat)

    if average_2d == 0:
        improvement = 0.0
    else:
        improvement = ((average_2d - average_1d) / average_2d) * 100

    return average_2d, average_1d, size * size, improvement
