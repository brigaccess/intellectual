from functools import reduce

def slide1_task1(limit):
    # definitely an oversight, but will do for the current formulation
    value = 2
    days = 1
    while value < limit:
        days += 1
        value *= 2
    return days

print(slide1_task1(1000))
print(slide1_task1(10000))

def slide1_task2(limit):
    # same oversight, same reasoning
    primes = [2]
    value = 3
    primes_sum = 2
    days = 1
    # Erathosphene's sieve, sort of
    while primes_sum < limit and value <= limit / 2:
        for p in primes:
            if value % p == 0:
                break
        else:
            primes.append(value)
            days += 1
            primes_sum += value
        value += 1
    return days

print(slide1_task2(1000))

def slide1_task3(start=10, increment_coefficient=1.15, limit=30):
    norm = start
    path = 0
    must_increase = True
    for i in range(limit):
        path += norm
        if must_increase:
            norm *= increment_coefficient
        must_increase = not must_increase
    return path

print(slide1_task3())

def slide1_task4(start=10, increment_coefficient=1.1, limit_path=100, trigger_norm=20):
    norm = start
    path = 0
    days = 0
    days_at_limit = None
    triggered_at_day = None
    while days_at_limit is None or triggered_at_day is None:
        if not days_at_limit and path >= limit_path:
            days_at_limit = days

        days += 1
        path += norm
        print(days, path, norm)
        if not triggered_at_day and norm >= trigger_norm:
            triggered_at_day = days

        norm *= increment_coefficient
    return days_at_limit, triggered_at_day

print(slide1_task4())

def slide2_task1(n, initial=[1, 1]):
    if n <= 0:
        return None
    if n <= len(initial):
        return initial[n - 1]
    else:
        buffer = list(initial)
        step = len(initial)
        while step < n:
            buffer.append(sum(buffer))
            buffer.pop(0)
            step += 1
        return buffer[-1]

print(slide2_task1(6))

def slide2_task2(n):
    return slide2_task1(n, initial=[1, 1, 1])

print(slide2_task2(6))

def slide2_task3(n):
    return [i ** 2 for i in range(1, n, 2)]

print(slide2_task3(10))

def slide2_task4(a, b):
    return sum(range(a, b + 1)), reduce(lambda x, y: x * y, range(a, b + 1))

print(slide2_task4(1, 10))

def slide2_task5(a, b):
    return [i for i in range(a, b + 1) if i % 2 == 0], [i for i in range(a, b + 1) if i % 2 != 0]

print(slide2_task5(1, 10))

def slide2_task6(l):
    return [i for i in l if i >= 0], [i for i in l if i < 0]

print(slide2_task6([1, -1, 2, -2, 3, -3, -342, 0, 468, -972]))

def slide3(w, h, c="*", thickness=1):
    required_width = thickness * 2
    required_height = thickness * 2
    if w < required_width:
        raise ValueError("Minimal required width for thickness {} is {}, but {} is given".format(
            thickness, required_width, w
        ))
    if h < required_height:
        raise ValueError("Minimal required height for thickness {} is {}, but {} is given".format(
            thickness, required_height, h
        ))
    inside_width = w - required_width
    inside_height = h - required_height
    for i in range(thickness):
        print(w * c)
    for i in range(inside_height):
        print((c * thickness) + (" " * inside_width) + (c * thickness))
    for i in range(thickness):
        print(w * c)

slide3(3, 3)
slide3(5, 5, c="!")
slide3(7, 7, thickness=2)