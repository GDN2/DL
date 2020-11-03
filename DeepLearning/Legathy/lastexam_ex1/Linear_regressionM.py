import numpy as np
import numerical_derivative_final as nd
import matplotlib.pyplot as plt

loaded_data = np.loadtxt('./Picoscope(Second Test)v3.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:,2].reshape(240,1)
t_data = loaded_data[:,3].reshape(240,1)
Time = [x for x in range(240)]
print(Time)


W = np.random.rand(1,1)
b = np.random.rand(1)
print("x_data =", x_data, "t_data =", t_data)
print("W =", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)

def loss_func(x,t):
    y = np.dot(x,W) + b

    return (np.sum((t-y)**2))/(len(x))


def error_val(x, t):
    y = np.dot(x, W) + b

    return (np.sum((t - y) ** 2)) / (len(x))

f = lambda x : loss_func(x_data, t_data)

learning_rate = 1e-3
print("Initial error value =", error_val(x_data, t_data), "Initial W =", W, "\n", ", b =", b)

for step in range(10001):
    W -= learning_rate * nd.numerical_derivative(f,W)
    b -= learning_rate * nd.numerical_derivative(f,b)
    if(step % 400 ==0):
        print("step =", step, "error value =", error_val(x_data,t_data), "W =", W, ", b =",b)

e = np.zeros_like(t_data)
predict_velocity = []
def predict():
    cnt = 0
    delta = 1e-7
    y = np.dot(x_data, W) + b
    for i in range(len(t_data)):
        error_rate = (t_data[i] - y[i]) / (t_data[i] + delta) * 100
        print(i,"예측속도: ",y[i],"실제속도: ",t_data[i],"오차율[%]", error_rate
              ,"W =",W, "b =", b )
        predict_velocity.append(y[i])
        if (error_rate<=100 and error_rate>=-100):
            e[i,0] =  error_rate
            cnt += 1
    print("평균오차율 =",  np.sum(e)/cnt, "%")
predict()
def predict2(x):
    y = np.dot(x, W) + b
    return y
print("토크가 8.88일때의 실제속도는 =", predict2([8.88]))
print("토크가 17.17일때의 실제속도는 =", predict2([17.17]))
print("토크가 6.55일때의 실제속도는 =", predict2([6.55]))



plt.grid()
plt.plot(Time, predict_velocity, Time, t_data)
plt.show()


