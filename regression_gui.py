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
_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False}
gen_VARS={'n_samples':100, 
         'n_features':1, 
         'n_informative':1, 
         'n_targets':1,
         'bias':0.0, 
         'tail_strength':0.5,
         'noise':10.0, 
         'shuffle':True, 
         'coef':True, 
         'random_state':None}

gen=RegGen()
print(gen.attributes)
gen.set_attr(gen_VARS)
print(gen.attributes)

plt.style.use('Solarize_Light2')

# Helper Functions


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


# \\  -------- PYSIMPLEGUI -------- //

AppFont = 'Any 16'
SliderFont = 'Any 14'
sg.theme('black')

# New layout with slider and padding
layout = [[sg.Canvas(key='figCanvas', background_color='#FDF6E3')],
          [sg.Text(text="sample_size :",
                   font=SliderFont,
                   background_color='#FDF6E3',
                   pad=((0, 0), (10, 0)),
                   text_color='Black'),
           sg.Slider(range=(4, 1000), orientation='h', size=(34, 20),
                     default_value=gen_VARS['n_samples'],
                     background_color='#FDF6E3',
                     text_color='Black',
                     key='n_samples',
                     enable_events=True),],
           [
           sg.Text('n_features', size =(15, 1)), sg.InputText(key='n_features'),
           sg.Text('n_informative', size =(15, 1)), sg.InputText(key='n_informative'),
           sg.Text('n_targets', size =(15, 1)), sg.InputText(key='n_targets')],
          [
          sg.Text('bias', size =(15, 1)), sg.InputText(key='bias'),
          sg.Text('noise', size =(15, 1)), sg.InputText(key='noise'),
          sg.Text('tail_strength', size =(15, 1)), sg.InputText(key='tail_strength')
          ],         
          [
          sg.Button('Resample',
                     font=AppFont,
                     pad=((4, 0), (10, 0))),
          sg.Button('Exit', font=AppFont, pad=((540, 0), (0, 0))),
          sg.Button('Save', font=AppFont, pad=((540, 10), (0, 0)))
          ]]

_VARS['window'] = sg.Window('Random Samples',
                            layout,
                            finalize=True,
                            resizable=True,
                            location=(100, 100),
                            element_justification="center",
                            background_color='#FDF6E3')

# \\  -------- PYSIMPLEGUI -------- //


# \\  -------- PYPLOT -------- //
df=pd.DataFrame()

def makeSynthData():
    data = gen.generate()   
    return data[0],data[1]

def plot(data):   
    fig, ax = plt.subplots()
    X,y = makeSynthData()
    data = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))    
    colors = sns.color_palette("husl", 300)    
    grouped = data.groupby('label') 
    # data.plot(ax=ax, kind='scatter', x='x', y='y')
    for key, group in grouped: 
        group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[1])    
    return fig
def drawChart(data):    
    
    _VARS['pltFig'] = plot(data)
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])


def updateChart(data):
    _VARS['fig_agg'].get_tk_widget().forget()
    plt.cla()
    # plt.clf()
    _VARS['pltFig'] = plot(data)
    # plt.plot(X, y, '.k')
    _VARS['fig_agg'] = draw_figure(
        _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])


# def updateData(val,data):
#     _VARS['n_samples'] = val
#     updateChart(data)
def updateGen(val,data,key) :
    gen_VARS[key]=val
    gen.set_attr({key:val})
    # print(gen.n_samples)
    updateChart(data)
# \\  -------- PYPLOT -------- //


drawChart(df)

# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Resample':
        updateChart(df)
    elif event == 'n_samples':
        updateGen(int(values['n_samples']),df,event)
    elif event == 'n_features':
        updateGen(int(values['n_features']),df,event)
    elif event == 'n_informative':
        updateGen(int(values['n_informative']),df,event) 
    elif event == 'n_targets':
        updateGen(int(values['n_targets']),df,event)
    elif event == 'bias':
        updateGen(int(values['bias']),df,event) 
    elif event == 'noise':
        updateGen(int(values['noise']),df,event)
    elif event == 'tail_strength':
        updateGen(int(values['tail_strength']),df,event)                            
    if event == 'Save':
        from os.path import exists

        if not(exists('./generated/')):
            os.mkdir('./generated/')
        out_str='./generated/sample_size_'+str(gen_VARS['n_samples'])+ \
        '_n_features_'+str(gen_VARS['features'])+'_centers_'+str(gen_VARS['centers'])+'.csv'
        gen_VARS['data'].to_csv(out_str)
        print("output saved to:\n",out_str)    
        break    
        # print(values)
        # print(int(values['-Slider-']))
    # elif event == '-Slider-':
_VARS['window'].close()