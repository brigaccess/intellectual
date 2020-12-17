def check(number):
    if len(number) < 4:
        return False
    if not number.isdigit():
        return False

    control = lambda x: x * 2 if x * 2 < 9 else x * 2 - 9
    offset = len(number) % 2

    digits = list(map(int, number))
    checksum_parts = [control(digits[i]) 
        if (i % 2) == offset else digits[i]
        for i in range(offset, len(digits))]
    return sum(checksum_parts) % 10 == 0
    

if __name__ == '__main__':
    while True:
        number = input("Для того, чтобы получить $$$, введите номер карты: ")
        number = number.replace(" ", "")
        if check(number):
            print("$$$")
            break
        else:
            print("Это не номер карты.")