# create a plotting class called Plotter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")


class Plotter:
    def __init__(self):
        pass

    def read_csv_file(self, filename):
        return pd.read_csv(filename, header=None, names=["x", "y", "p(x)"])
    
    def plot_kde(self, df,hist=False,rug=False,kde=False):
        
        plt.figure(figsize=(12, 8))
        
        if hist: 
            sns.histplot(data=df, x='y', kde=False, stat='density', bins=100, color='blue', label='Histogram')
        
        if rug:
            sns.rugplot(df.x[::100], height=0.02, color='black',label='N = 8000 Samples')
        if kde:
            sns.lineplot(data=df, x='x', y='p(x)',color='red', label='KDE ')
            
        plt.ylabel('p(x)')
        plt.legend()
        plt.show()