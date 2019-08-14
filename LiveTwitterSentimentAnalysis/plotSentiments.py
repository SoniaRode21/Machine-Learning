import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use("ggplot")

figure= plt.figure()
ax1 = figure.add_subplot(1,1,1)
def custom_animate_fun(i):
    pullData = open("twitterSentiments.txt","r").read()
    lines = pullData.split('\n')

    xlist,ylist = [],[]
    x,y= 0,0
    for l in lines[-100:]:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1
        xlist.append(x)
        ylist.append(y)

    ax1.clear()
    ax1.plot(xlist,ylist)
ani = animation.FuncAnimation(figure, custom_animate_fun, interval=1000)
plt.show()
