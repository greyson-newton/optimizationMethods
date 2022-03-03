import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from RegGen import *
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
import os
from RegGen import *
# VARS CONSTS:
# Upgraded n_samples to global...

class gui:
    def __init__(self):

        self._VARS = {'window': False,
                 'fig_agg': False,
                 'pltFig': False}
        self.gen_VARS={'n_samples':100, 
                 'n_features':1, 
                 'n_informative':1, 
                 'n_targets':1,
                 'bias':0.0, 
                 'tail_strength':0.5,
                 'noise':10.0, 
                 'shuffle':True, 
                 'coef':True, 
                 'random_state':True,
                 'data':None,}

        plt.style.use('Solarize_Light2')
        AppFont = 'Any 16'
        SliderFont = 'Any 14'
        sg.theme('black')
        self.data=pd.DataFrame()
        self.gen=RegGen()
        # print(gen.attributes)
        self.gen.set_attr(self.gen_VARS)
        # print(gen.attributes)

        self.layout = [[sg.Canvas(key='figCanvas', background_color='#FDF6E3')],
                  [sg.Text(text="sample_size :",
                           font=SliderFont,
                           background_color='#FDF6E3',
                           pad=((0, 0), (10, 0)),
                           text_color='Black'),
                   sg.Slider(range=(4, 1000), orientation='h', size=(34, 20),
                             default_value=self.gen_VARS['n_samples'],
                             background_color='#FDF6E3',
                             text_color='Black',
                             key='n_samples',
                             enable_events=True),],
                   [
                   sg.Text('n_features', size =(15, 1)), self.rs((1,10),(15,20),'n_features'),
                   sg.Text('n_informative', size =(15, 1)), self.rs((1,10),(15,20),'n_informative'),
                   sg.Text('n_targets', size =(15, 1)), self.rs((1,10),(15,20),'n_targets'),],
                  [
                  sg.Text('bias', size =(15, 1)), self.rs((1,100),(15,20),'bias'),
                  sg.Text('noise', size =(15, 1)), self.rs((1,100),(15,20),'noise') ,
                  sg.Text('tail_strength', size =(15, 1)), self.rs((1,100),(15,20),'tail_strength'),],         
                  [
                  sg.Button('Resample',
                             font=AppFont,
                             pad=((4, 0), (10, 0))),
                  sg.Button('Exit', font=AppFont, pad=((540, 0), (0, 0))),
                  sg.Button('Save', font=AppFont, pad=((540, 10), (0, 0)))
                  ]]  
        self._VARS['window'] = sg.Window('Random Samples',
                                    self.layout,
                                    finalize=True,
                                    resizable=True,
                                    location=(100, 100),
                                    element_justification="center",
                                    background_color='#FDF6E3')                    
    def rs(self,rg,size,name):
        return sg.Slider(range=rg, orientation='h', size=size,
                             default_value=self.gen_VARS[name],
                             background_color='#FDF6E3',
                             text_color='Black',
                             key=name,
                             enable_events=True) 
    def draw_figure(self,canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg


    def makeSynthData(self):
        data = self.gen.generate()   
        print(data)
        # print(len(data[0]))
        return data

    def plot(self):   
        fig, ax = plt.subplots()
        self.data = self.makeSynthData()
        # print('X',X.size,'y',y.size)
        # data = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))    
        
        colors = sns.color_palette("husl", 10)    
        # grouped = data.groupby('label') 
        cols = [ c for c in self.data.columns if 'x' in c]
        for i,c in enumerate(cols):
            self.data.plot(ax=ax, kind='scatter', x=c, y='y',color=colors[i])
        # for key, group in grouped: 
        #     group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[1])    
        return fig
    def drawChart(self):    
        
        self._VARS['pltFig'] = self.plot()
        self._VARS['fig_agg'] = self.draw_figure(
            self._VARS['window']['figCanvas'].TKCanvas, self._VARS['pltFig'])


    def updateChart(self):
        self._VARS['fig_agg'].get_tk_widget().forget()
        plt.cla()
        # plt.clf()
        self._VARS['pltFig'] = self.plot()
        # plt.plot(X, y, '.k')
        self._VARS['fig_agg'] = self.draw_figure(
            self._VARS['window']['figCanvas'].TKCanvas, self._VARS['pltFig'])

    def updateGen(self,val,key) :
        self.gen_VARS[key]=val
        self.gen.set_attr({key:val})
        # print(gen.n_samples)
        self.updateChart()
# \\  -------- PYPLOT -------- //


app = gui()
app.drawChart()

# MAIN LOOP
while True:
    event, values = app._VARS['window'].read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Resample':
        app.updateChart()
    elif event == 'n_samples':
        app.updateGen(int(values['n_samples']),event)
    elif event == 'n_features':
        app.updateGen(int(values['n_features']),event)
    elif event == 'n_informative':
        app.updateGen(int(values['n_informative']),event) 
    elif event == 'n_targets':
        app.updateGen(int(values['n_targets']),event)
    elif event == 'bias':
        app.updateGen(int(values['bias']),event) 
    elif event == 'noise':
        app.updateGen(int(values['noise']),event)
    elif event == 'tail_strength':
        app.updateGen(int(values['tail_strength']),event)                            
    if event == 'Save':
        from os.path import exists

        if not(exists('./generated/')):
            os.mkdir('./generated/')
        out_str='./generated/sample_size_'+str(app.gen_VARS['n_samples'])+'.csv'
        app.data.to_csv(out_str)
        print("output saved to:\n",out_str)    
        break    
        # print(values)
        # print(int(values['-Slider-']))
    # elif event == '-Slider-':
app._VARS['window'].close()