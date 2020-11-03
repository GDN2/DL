import numpy as np

x_data = np.array([0,0,0,1,1,0,1,1]).reshape(-1,2)
and_t_data = np.array([0,0,0,1]).reshape(-1,1)
or_t_data = np.array([0,1,1,1]).reshape(-1,1)
nand_t_data = np.array([1,1,1,0]).reshape(-1,1)
xor_t_data = np.array([0,1,1,0]).reshape(-1,1)
W2 = np.random.rand(2,6)
b2 = np.random.rand(6)
W3 = np.random.rand(6,1)
b3 = np.random.rand(1)
learning_rate = 1e-2
delta_x = 1e-7
steps = 40001

def sigmoid(z):
     return (1/(1+np.exp(-z))) # exp에는 delta_x필요 없음

def loss_func(x_data,t_data):

    z2 = np.dot(x_data, W2) + b2 # a2에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3 # a3에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
    y = a3 = sigmoid(z3)
    E = -np.sum(t_data * np.log(y + delta_x) + (1 - t_data) * np.log(1 - y + delta_x))
    return E


def error_val(x_data, t_data):

    z2 = np.dot(x_data, W2) + b2  # a2에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3  # a3에다가 붙이면 안됨 y값이 1보다 커지고 log에 -들어가서 에러
    y = a3 = sigmoid(z3)
    E = -np.sum(t_data * np.log(y + delta_x) + (1 - t_data) * np.log(1 - y + delta_x))
    return E

def numerical_derivative(f,x):

    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
    while not it.finished:
        idx = it.multi_index
            
        tmp_val = x[idx]
        x[idx] = float(tmp_val)+delta_x
        f1 = f(x)
        x[idx] = float(tmp_val)-delta_x
        f2 = f(x)
           
        grad[idx] = (f1-f2)/(2*delta_x)# 분자에 괄호
            
        x[idx] = tmp_val
        it.iternext()
        
    return grad #return indent while에 맞추기

def predict(x):
    z2 = np.dot(x, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    y = a3 = sigmoid(z3)
    result = 0
    if y >= 0.5:
        result = 1
        return y, result # return indent위치 주의하고 마지막에 한 번 더 return 하지 말기
    elif y < 0.5:
        result = 0
        return y, result

# 실수로 값 받기)

f = lambda x : loss_func(x_data,xor_t_data)
print( "Initial W2 =", W2, ", \nb2 =", b2, ",\nW3 =", W3, ", \nb3 =", b3, "\nInitial error value =", error_val(x_data, and_t_data))
for step in range(steps):

    W2 -= learning_rate * numerical_derivative(f, W2)
    b2 -= learning_rate * numerical_derivative(f, b2)
    W3 -= learning_rate * numerical_derivative(f, W3)
    b3 -= learning_rate * numerical_derivative(f, b3)

    if (step % 400 == 0):
         print("step =", step, "error_value =", error_val(x_data, and_t_data))

for data in x_data:
    (real_val, logical_val) = predict(data)
    print("rela_val =", real_val, "logical_val =", logical_val)

