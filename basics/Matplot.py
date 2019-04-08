import numpy as np 
import random as ran
# pip3 install matplotlib

from matplotlib import pyplot as plt 

x = np.array([160,79,200,9,-10,2,6,4,5,121])
y = np.array([-11,9,100,55,77,99,15,88,7,10])

#plt.title('Sample')
#plt.xlabel('X Axis')
#plt.ylabel('Y Axis')
#plt.plot(x,y)
# rolling objects
#plt.plot(x,y,'ro')
#plt.show()

# bar plot
#plt.bar(x,y)
#plt.xlabel('X Axis')
#plt.ylabel('Y Axis')
#plt.show()

# height : 2
# width : 1
# active index : 1
#plt.subplot(2,1,1)
#plt.plot(x,x)
#plt.title('Plot One')

#plt.subplot(2,1,2)
#plt.plot(x,y)
#plt.title('Plot Two')

# Pie chart
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.10, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels,shadow = True,
        startangle=90)
ax1.axis('equal') 

plt.show()



