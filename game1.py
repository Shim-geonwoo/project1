import random

#숫자 랜덤 생성
def generate_number():
    return [random.randint(0, 9) for _ in range(4)]

#스트라이크 및 볼
def hit_and_run(secret_number, my_num):
    strikes = 0
    balls = 0
    for i in range(len(my_num)):
        if my_num[i] == secret_number[i]:
            strikes += 1
        elif my_num[i] in secret_number:
            balls += 1

    return strikes, balls

# 게임 실행
def play_game():
    print("4자리 숫자야구 게임을 시작합니다. 0에서 9까지의 숫자 중 4개를 추측하세요.")
    secret_number = generate_number()
    attempts = 0
    print(secret_number)

    while True:
        my_num = input("4자리 숫자를 입력하세요: ")
        
        try:
            my_num = [int(x) for x in my_num]
        except ValueError:
            print("잘못된 입력입니다. 0에서 9까지의 숫자 중 4개를 입력하세요.")
            continue
        
        if len(my_num) != 4 or any(x < 0 or x > 9 for x in my_num):
            print("잘못된 입력입니다. 0에서 9까지의 숫자 중 4개를 입력하세요.")
            continue

        attempts += 1
        strikes, balls = hit_and_run(secret_number, my_num)

        if my_num == secret_number:
            print(f"축하합니다! {attempts}번만에 숫자를 맞추셨습니다.")
            break
        else:
            print(f"{strikes} 스트라이크, {balls} 볼입니다.")

if __name__ == "__main__":
    play_game()