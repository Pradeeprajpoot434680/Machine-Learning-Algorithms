import numpy as np
import matplotlib.pyplot as plt

#f(x, y) = 3x^2 + 2y^2 + xy

def f(x, y):
    return 3 * x**2 + 2 * y**2 + x * y


def gradients(x, y):
    df_dx = 6 * x + y
    df_dy = 4 * y + x
    return df_dx, df_dy




def gradient_descent(x, y, learning_rate=0.05, iterations=100):

    loss_history = []

    for i in range(iterations):

       
        df_dx, df_dy = gradients(x, y)

 
        x = x - learning_rate * df_dx
        y = y - learning_rate * df_dy

        
        loss = f(x, y)
        loss_history.append(loss)

        
        print(f"Iteration {i+1}: x = {x:.6f}, y = {y:.6f}, Loss = {loss:.6f}")

    return x, y, loss_history



x_init = 5
y_init = 3

final_x, final_y, losses = gradient_descent(
    x_init,
    y_init,
    learning_rate=0.05,
    iterations=100
)



print("\nFinal Optimized Values:")
print("x =", final_x)
print("y =", final_y)


plt.plot(range(1, 101), losses)
plt.xlabel("Iterations")
plt.ylabel("Loss Function Value")
plt.title("Loss Function vs Iterations")
plt.grid(True)
plt.show()