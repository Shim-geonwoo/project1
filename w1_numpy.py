import numpy as np

# ar = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(type(ar))

# data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# answer = []
# for di in data:
#     answer.append(2 * di)

# print(answer)


# x = np.array(data)
# print(2 * x)


# 2차원 배열
# c = np.array([[0, 1, 2], [3, 4, 5]])
# len(c) #행의 개수
# len(c[0]) #열의 개수


# print(len(c))
# print(len(c[1]))

# c = np.array([[0, 1, 2], [3, 4, 5]])
# print(c)

#배열의 차원과 크기
#배열의 차원은 d.nim
#배열의 크기는 shape

#배열의 인덱스
#1차원 배열은 리스트의 인덱스와 동일
#2차원 배열은 a[0,1] 첫번째 행의, 두번째 열

#배열 슬라이싱
#0:? 랑 ,를 써서 슬라이싱가능


#연습문제 2

# m = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
# print(m.ndim)
# print(m.shape)

# print(m)
# print(m[1,2])
# print(m[2,4])
# print(m[1,1:3])
# print(m[1:3, 2])
# print(m[0:2, 3:5])


#배열인덱싱
#불리언 배열방식과 정수 배열방식
# a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# idx = np.array([True, False, True, False, True, False, True, False, True, False])
# a[idx]
# print(a[idx])



#연습문제 3
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#              11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# idx = np.array([False, False, True, False, False, True, False, False, True, False, False, True,
#  False, False, True, False, False, True, False, False])

# #1.
# print(x[x%3 == 0])

# #2.
# print(x[x%4 == 1])

# #3.
# print(x([x % 4 == 0] and [x % 3 == 1]))

# -------------------- #

#넘파이의 자료형
#dtpye로 int형인지 float형인지 확인가능


#배열생성 명령

# a = np.zero(5)
# b = np.zero((2,3))
# c = np.zeros((5,2), dtype = "i")
# c = np.zeros(5, dtype = "U4")

# d[0] = "abc"
# d[1] = "abcd"
# d[2] = "abcde"

# e = np.ones((2, 3, 4), dtype="i8")
# print(e)

# f = np.ones_like(b, dtype="f")
# print(f)

# g = np.empty((4, 3))
# print(g)

# #수열만들기
# np.arange(10)
# np.arange(1,18,6)

# #선형 혹은 로그 분할
# np.linspace(0, 100, 5) #선형
# np.logspace(0.1, 1, 10) #로그

# #행 열 바꾸기 transpose
# A = np.array([[1, 2, 3], [4, 5, 6]])
# print(A.T)

# #형태 바꾸기 reshape
# X = np.arange(10)
# Y = X.reshape(2,5)

# -------------------- #
#벡터와 연산
# x = np.arange(1, 10001)
# y = np.arange(10001, 20001)

# # z = np.zeros_like(x)
# # for i in range(10000):
# #     z[i] = x[i] + y[i]
# # z[:10]

# z = x + y
# print(z[:10])

# a = np.array([1, 2, 3, 4])
# b = np.array([4, 2, 2, 4])
# c = np.array([1, 2, 3, 4])

# a == b
# a >= b

# np.all(a == b) #배열의 모든 원소가 같은지 전수비교
# np.all(a == c)

# 곱셈
# x = np.arange(10)
# 100 * x

# x = np.arange(12).reshape(3, 4)
# x

#브로드캐스팅(스칼라를 벡터크기로 확장시켜 연산)

# x = np.arange(5)
# y = np.ones_like(x)
# x + y

# x = np.vstack([range(7)[i:i + 3] for i in range(5)])
# y = np.arange(5)[:, np.newaxis]
# x + y

#차원 축소 연산
# a = np.arange(30)
# a = a.reshape(5,6)
# print(a)
# print(a.sum())
# print(a.sum(axis=0))
# print(a.max(axis=0))
# print(a.mean(axis=1))
# print(a.min(axis=1))

#정렬
# a = np.array([[4,  3,  5,  7],
#               [1, 12, 11,  9],
#               [2, 15,  1, 14]])

# np.sort(a)
# np.sort(a, axis=0)
# a.sort(axis=1)

# a = np.array([42, 38, 12, 25])
# j = np.argsort(a)

#기술통계 descriptive stastic
#넘파이에서는 데이터를 1차원 배열로 구성
# x = np.array([18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
#               2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13])

# print(len(x)) #데이터 개수
# print(np.mean(x)) #평균
# print(np.var(x)) #분산
# print(np.std(x)) #std(x) #표준편차
# print(np.max(x)) #max(x) #최대값
# print(np.min(x)) #최소값
# print(np.median(x)) #중앙값
#사분위 수
# print(np.median(x)) #percentile(x, 0) #0, 25, 50, 75, 100

# # -------------------- #
# #시드 설정하기
# print(np.random.seed(0))
# print(np.random.rand(5))

# x = np.arange(10)
# print(np.random.shuffle(x))

#데이터 샘플링
# numpy.random.choice(a, size=None, replace=True, p=None)
# a : 배열이면 원래의 데이터, 정수이면 arange(a) 명령으로 데이터 생성
# size : 정수. 샘플 숫자
# replace : 불리언. True이면 한번 선택한 데이터를 다시 선택 가능
# p : 배열. 각 데이터가 선택될 수 있는 확률
# np.random.choice(5, 3, replace=False)
# np.random.choice(5, 10) #replace = true
# np.random.choice(5, 10, p=[0.1, 0, 0.3, 0.6, 0])

#난수생성
# rand: 0부터 1사이의 균일 분포
# randn: 표준 정규 분포
# randint: 균일 분포의 정수 난수

# np.random.rand(10)
# np.random.rand(3, 5)
# print(np.random.rand(2, 3, 5))
# randn도 동일하게 사용

# numpy.random.randint(low, high=None, size=None)
# np.random.randint(10, size=10)
# np.random.randint(10, 20, size=10)
# np.random.randint(10, 20, size=(3, 5))

#동전던지기, 주사위던지기
# print(np.random.randint(0, 2, size=10))
# dice = np.random.randint(1, 7, size=100)
# print(np.mean(dice))

#주식
R = np.random.randn(10)
stock = []
for i in range(10):
    price = 10000
    price = price * (R[i-1]/100 + 1)
    stock.append(price)
    

stock = np.array(stock)
print(stock)


#정수 데이터 카운팅
np.unique([11, 11, 2, 2, 34, 34])
a = np.array(['a', 'b', 'b', 'c', 'a'])
index, count = np.unique(a, return_counts=True)
index
count
np.bincount([1, 1, 2, 2, 2, 3], minlength=6)
