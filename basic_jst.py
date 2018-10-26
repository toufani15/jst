xx = [2., 3., 2., 4.]
x_inp = [1.0, 2.0, 3.0]
y_inp = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

def grad(x,y):
    return 2*x*(x*w - y)

for epoch in range(10):
    for x_val, y_val in zip(x_inp, y_inp):
        print(x_val)
