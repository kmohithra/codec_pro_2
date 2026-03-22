import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
days=np.array(range(1, 366)).reshape(-1, 1)  
temp=20+10*np.sin(2*np.pi*days/365)+np.random.normal(0,2,days.shape)

df=pd.DataFrame({'Day':days.flatten(),'Temperature':temp.flatten()})

X=df[['Day']]
y=df['Temperature']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

predictions=model.predict(X_test)
mse=mean_squared_error(y_test,predictions)
print(f"Model Mean Squared Error:{mse:.2f}")

plt.scatter(X, y, color='blue',label='Actual Data',s=10)
plt.plot(X,model.predict(X),color='red',label='Trend Line')
plt.title('Temperature Prediction Trend')
plt.xlabel('Day of the Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()