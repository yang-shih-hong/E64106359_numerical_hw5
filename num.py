#第二題程式碼
import numpy as np

def f(t, u):
    u1, u2 = u
    du1_dt = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2_dt = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1_dt, du2_dt])

def exact_solution(t):
    u1 = 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)
    u2 = -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)
    return np.array([u1, u2])

def RK4(f, u0, t0, t_end, h):
    t_values = np.arange(t0, t_end+h, h)
    u_values = np.zeros((len(t_values), len(u0)))
    u_values[0] = u0
    
    for i in range(len(t_values)-1):
        t = t_values[i]
        u = u_values[i]
        k1 = h * f(t, u)
        k2 = h * f(t + h/2, u + k1/2)
        k3 = h * f(t + h/2, u + k2/2)
        k4 = h * f(t + h, u + k3)
        u_values[i+1] = u + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t_values, u_values

# 初始條件
u0 = np.array([4/3, 2/3])

# 設定參數
t0 = 0
t_end = 1  # 算到多少
h_list = [0.05, 0.1]

for h in h_list:
    t_values, u_values = RK4(f, u0, t0, t_end, h)
    exact_values = np.array([exact_solution(t) for t in t_values])
    error = np.abs(u_values - exact_values)

    print(f"\n步長 h = {h}")
    for i in range(len(t_values)):
        print(f"t = {t_values[i]:.2f}, RK4 = {u_values[i]}, Exact = {exact_values[i]}, Error = {error[i]}")
