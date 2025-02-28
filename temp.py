import numpy as np
import torch
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui





# x = [0,1,2,3,4,5,6,7,8,9]
# y = [11,23,5,123,75,66,13,34,77,9]


# plt = pg.plot()
# plt.showGrid(x=True,y=True)
# plt.addLegend()
# plt.setLable('left', 'Rewards', units = 'y')

# plt.setLable('bottom', 'Episodes', units = 'e')



# plt.setXRange(0,10)

# plt.setYRange(0,100)

# plt.setWindowTitle("Title")

# line1 = plt.plot(x,y,pen = 'green', symbor= 'x', symbolPen = 'red', symbolBrush = 0.2, name = 'reward')



# # main method
# if __name__ == '__main__':
     
#     # importing system
#     import sys
     
#     # Start Qt event loop unless running in interactive mode or using 
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()







import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from random import randint
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Temperature vs time dynamic plot
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")
        pen = pg.mkPen(color=(255, 0, 0))
        self.plot_graph.setTitle("Losses per episode", color="b", size="20pt")
        styles = {"color": "red", "font-size": "20px"}
        self.plot_graph.setLabel("left", "Losses", **styles)
        self.plot_graph.setLabel("bottom", "Episodes", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(-100, 100)
        self.episodes = [0,1,2,3,4,]
        self.actor_losses= [1,4,7,11,13]
        self.critic_losses= [2,-2,-5,12,33]
        self.temp_losses= [1,2,3,4,5]
        
        # Get a line reference
        self.actor_pen = pg.mkPen(color= (255,0,0))
        self.actor_line = self.plot_graph.plot(
            self.episodes,
            self.actor_losses,
            name="Actor losses",
            pen=self.actor_pen,
            symbol="d",
            symbolSize=5,
            symbolBrush="b",
        )
        self.critic_pen = pg.mkPen(color= (0,255,0))
        self.critic_line = self.plot_graph.plot(
            self.episodes,
            self.critic_losses,
            name="Critic losses",
            pen=self.critic_pen,
            symbol="d",
            symbolSize=5,
            symbolBrush="g",
        )
        self.temp_pen = pg.mkPen(color= (0,0,255))
        self.temp_line = self.plot_graph.plot(
            self.episodes,
            self.temp_losses,
            name="Temp losses",
            pen=self.temp_pen,
            symbol="d",
            symbolSize=5,
            symbolBrush="r",
        )



        ### Add a timer to simulate new temperature measurements
        self.timer = QtCore.QTimer()
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.update_plot(self.episodes,self.actor_losses,self.critic_losses,self.temp_losses))
        self.timer.start()
        #self.plot_all(self.episodes,self.actor_losses,self.critic_losses,self.temp_losses)

    def update_plot(self,actor_loss,critic_loss,temp_loss):
        #self.time = self.time[1:]
        #self.time.append(self.time[-1] + 1)
        #self.temperature = self.temperature[1:]
        #self.temperature.append(randint(20, 40))
        #self.line.setData(self.time, self.temperature)
        self.episodes.append(len(self.episodes))
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.temp_losses.append(temp_loss)

        self.plot_all(self.episodes, self.actor_losses, self.critic_losses, self.temp_losses)




    def loss_plot(self,name,episodes,actor_losses,pen,brush):
        self.plot_graph.plot(
            episodes,
            actor_losses,
            name=name,
            pen=pen,
            symbol="d",
            symbolSize=5,
            symbolBrush=brush,
        )

    def plot_all(self,episodes, actor_losses, critic_losses, temp_losses):
        
        self.loss_plot('Actor losses', episodes, actor_losses, self.actor_pen,'b')
        self.loss_plot('Critic losses', episodes, critic_losses, self.critic_pen,'g')
        self.loss_plot('Temp losses', episodes, temp_losses, self.temp_pen,'r')



app = QtWidgets.QApplication([])
main = MainWindow()
main.show()
app.exec()

main.update_plot(-50,22,45)

