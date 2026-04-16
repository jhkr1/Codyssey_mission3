import time

EPSILON = 1e-9
REPEAT_COUNT = 10


def normalize_label(raw_label):
    """여러 방식으로 적힌 라벨을 프로그램 내부 표준 이름으로 바꾼다.

    data.json에서는 Cross가 '+' 또는 'cross'로 들어올 수 있다.
    이렇게 표현이 달라도 같은 뜻이면 같은 이름으로 맞춰야 PASS/FAIL 비교가 안정적이다.
    """
    if not isinstance(raw_label, str):
        return None

    mapping = {
        "+": "Cross",
        "cross": "Cross",
        "x": "X",
    }
    return mapping.get(raw_label.strip().lower())


def format_score(value):
    """MAC 점수를 화면에 보기 좋게 출력할 문자열로 바꾼다.

    계산에 실패한 경우에는 숫자가 없으므로 'N/A'를 보여 준다.
    정상 점수는 부동소수점 차이를 확인할 수 있도록 소수점 아래를 넉넉히 출력한다.
    """
    if value is None:
        return "N/A"
    return f"{value:.16f}"


def validate_generator_size(size):
    """자동 패턴 생성에 사용할 크기가 올바른지 확인한다.

    Cross와 X 패턴은 가운데 칸이 있어야 자연스럽기 때문에 홀수 크기만 허용한다.
    예를 들어 3x3, 5x5, 13x13은 가능하지만 4x4는 가운데 한 칸이 없다.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("패턴 생성기는 1 이상의 홀수 크기만 지원합니다.")


def create_empty_matrix(size):
    """모든 값이 0.0인 size x size 2차원 배열을 만든다."""
    matrix = []
    for _ in range(size):
        row = []
        for _ in range(size):
            row.append(0.0)
        matrix.append(row)
    return matrix


def generate_cross_pattern(size):
    """가운데 행과 가운데 열이 1인 Cross 패턴을 자동 생성한다."""
    validate_generator_size(size)

    result = create_empty_matrix(size)
    middle = size // 2

    for row in range(size):
        for col in range(size):
            if row == middle or col == middle:
                result[row][col] = 1.0

    return result


def generate_x_pattern(size):
    """두 대각선이 1인 X 패턴을 자동 생성한다."""
    validate_generator_size(size)

    result = create_empty_matrix(size)

    for row in range(size):
        for col in range(size):
            if row == col or row + col == size - 1:
                result[row][col] = 1.0

    return result


def generate_pattern(label, size):
    """라벨 이름에 맞는 패턴 생성 함수를 골라 실행한다."""
    if label == "Cross":
        return generate_cross_pattern(size)
    if label == "X":
        return generate_x_pattern(size)
    raise ValueError(f"알 수 없는 패턴 라벨입니다: {label}")


def flatten_matrix(matrix):
    """2차원 배열을 한 줄짜리 1차원 배열로 바꾼다.

    보너스 과제의 1D 최적화에서 사용한다.
    행을 위에서 아래로 읽고, 각 행의 값을 왼쪽에서 오른쪽으로 이어 붙인다.
    """
    flat = []
    for row in matrix:
        for value in row:
            flat.append(value)
    return flat


def validate_matrix(raw_matrix, expected_size=None, name="matrix"):
    """외부에서 들어온 값이 올바른 정사각형 숫자 배열인지 검사한다.

    JSON 데이터처럼 외부 입력은 모양이 틀릴 수 있으므로 바로 계산에 쓰면 위험하다.
    이 함수는 2차원 배열인지, 정사각형인지, 숫자로만 구성됐는지 확인한 뒤 float 행렬로 돌려준다.
    """
    if not isinstance(raw_matrix, list) or not raw_matrix:
        raise ValueError(f"{name}는 비어 있지 않은 2차원 배열이어야 합니다.")

    size = len(raw_matrix)
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{name}의 행 수가 {expected_size}가 아닙니다.")

    matrix = []
    for row_index, raw_row in enumerate(raw_matrix, start=1):
        if not isinstance(raw_row, list):
            raise ValueError(f"{name}의 {row_index}행이 배열 형식이 아닙니다.")

        if len(raw_row) != size:
            raise ValueError(f"{name}는 정사각형 2차원 배열이어야 합니다.")

        if expected_size is not None and len(raw_row) != expected_size:
            raise ValueError(f"{name}의 열 수가 {expected_size}가 아닙니다.")

        row = []
        for col_index, value in enumerate(raw_row, start=1):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}의 {row_index}행 {col_index}열 값은 숫자여야 합니다.")
            row.append(float(value))
        matrix.append(row)

    return matrix


def calculate_mac(pattern, matrix_filter):
    """2차원 패턴과 필터를 같은 위치끼리 곱한 뒤 모두 더해 MAC 점수를 구한다.

    점수가 높을수록 입력 패턴이 해당 필터와 더 비슷하다고 해석한다.
    NumPy 같은 외부 라이브러리 없이 이중 반복문으로 직접 계산한다.
    """
    if len(pattern) != len(matrix_filter):
        raise ValueError("패턴과 필터의 크기가 다릅니다.")

    total = 0.0
    size = len(pattern)

    for row in range(size):
        if len(pattern[row]) != size:
            raise ValueError("패턴이 정사각형 2차원 배열이 아닙니다.")

        if len(matrix_filter[row]) != size:
            raise ValueError("필터가 정사각형 2차원 배열이 아닙니다.")

        for col in range(size):
            total += pattern[row][col] * matrix_filter[row][col]

    return total


def calculate_mac_flat(pattern, matrix_filter):
    """1차원으로 펼친 패턴과 필터의 MAC 점수를 계산한다.

    계산 원리는 calculate_mac()과 같지만, 행/열 인덱스 대신 한 개의 index로 접근한다.
    2차원 배열보다 접근 과정이 단순해서 성능 비교용으로 사용한다.
    """
    if len(pattern) != len(matrix_filter):
        raise ValueError("1차원 패턴과 필터의 길이가 다릅니다.")

    total = 0.0
    for index in range(len(pattern)):
        total += pattern[index] * matrix_filter[index]

    return total


def average_mac_ms(pattern, matrix_filter, repeat=REPEAT_COUNT):
    """2차원 MAC 연산을 여러 번 실행하고 평균 실행 시간을 ms 단위로 구한다.

    한 번만 측정하면 컴퓨터 상태에 따라 값이 흔들릴 수 있다.
    그래서 같은 작업을 repeat번 반복한 뒤 평균을 내어 성능 표에 사용한다.
    """
    durations = []

    for _ in range(repeat):
        start = time.perf_counter()
        calculate_mac(pattern, matrix_filter)
        end = time.perf_counter()
        durations.append((end - start) * 1000)

    return sum(durations) / len(durations)


def average_mac_flat_ms(pattern, matrix_filter, repeat=REPEAT_COUNT):
    """1차원 MAC 연산을 여러 번 실행하고 평균 실행 시간을 ms 단위로 구한다."""
    durations = []

    for _ in range(repeat):
        start = time.perf_counter()
        calculate_mac_flat(pattern, matrix_filter)
        end = time.perf_counter()
        durations.append((end - start) * 1000)

    return sum(durations) / len(durations)


def average_two_filter_ms(pattern, filter_a, filter_b, repeat=REPEAT_COUNT):
    """사용자 입력 모드에서 필터 2개의 MAC 연산 평균 시간을 구한다."""
    durations = []

    for _ in range(repeat):
        start = time.perf_counter()
        calculate_mac(pattern, filter_a)
        calculate_mac(pattern, filter_b)
        end = time.perf_counter()
        durations.append((end - start) * 1000)

    return sum(durations) / len(durations)


def decide_label(score_cross, score_x, epsilon=EPSILON):
    """Cross 점수와 X 점수를 비교해 최종 라벨을 결정한다.

    두 점수 차이가 epsilon보다 작으면 사실상 동점으로 보고 UNDECIDED를 반환한다.
    이 정책은 부동소수점의 아주 작은 오차 때문에 억지 판정이 나는 일을 막는다.
    """
    if abs(score_cross - score_x) < epsilon:
        return "UNDECIDED"
    if score_cross > score_x:
        return "Cross"
    return "X"


def calculate_two_scores(pattern, cross_filter, x_filter):
    """하나의 패턴에 Cross 필터와 X 필터를 각각 적용해 두 점수를 함께 구한다."""
    score_cross = calculate_mac(pattern, cross_filter)
    score_x = calculate_mac(pattern, x_filter)
    return score_cross, score_x


def benchmark_mac(size, repeat=REPEAT_COUNT):
    """주어진 크기에서 2D MAC과 1D MAC의 평균 시간을 비교한다.

    같은 Cross 패턴을 사용해 2차원 계산과 1차원 계산을 각각 측정한다.
    반환값에는 평균 시간, 연산 칸 수(N²), 1D 방식의 개선율이 들어 있다.
    """
    pattern_2d = generate_cross_pattern(size)
    filter_2d = generate_cross_pattern(size)
    pattern_1d = flatten_matrix(pattern_2d)
    filter_1d = flatten_matrix(filter_2d)

    average_2d = average_mac_ms(pattern_2d, filter_2d, repeat)
    average_1d = average_mac_flat_ms(pattern_1d, filter_1d, repeat)

    if average_2d == 0:
        improvement = 0.0
    else:
        improvement = ((average_2d - average_1d) / average_2d) * 100

    return average_2d, average_1d, size * size, improvement
