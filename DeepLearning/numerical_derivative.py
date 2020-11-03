
import numpy as np

class regression:
    def __init__(self, delta_x=1e-4):
        self.delta_x = delta_x
    
    def numerical_derivative(self,f,x):
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            
            tmp_val = x[idx]
            x[idx] = float(tmp_val)+self.delta_x
            f1 = f(x)
            x[idx] = float(tmp_val)-self.delta_x
            f2 = f(x)
           
            grad[idx] = (f1-f2)/(2*self.delta_x)# 분자에 괄호
            
            x[idx] = tmp_val
            it.iternext()
        
        return grad #return indent while에 맞추기
# 실수로 값 받기
'''def my_func(W): #연습코드
    x = W[0,]
    y = W[1,]

    return 2*x +x*y
f = lambda W : my_func(W)
W = np.array([1.0,2.0])
obj = regression()
gradian = obj.numerical_derivative(f,W)
print(gradian)'''

