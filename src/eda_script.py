# author: Lennon Au-Yeung
# date: 2022-11-25


""" Creates exploratory data analysis figures from the preprocessed training data of the maternal health risk dataset

Usage: src/eda_script.py --data_location=<data_location> --output_location=<output_location>

Options:
--data_location=<data_location>    Location of the data to be used for eda
--output_location=<output_location>  Location to output the visulisations
"""

from docopt import docopt
import altair as alt
import pandas as pd
import os
from altair_saver import save
import dataframe_image as dfi
import vl_convert as vlc
from sklearn.model_selection import train_test_split

opt = docopt(__doc__) 

def display(i,train_df):
    '''
    Return altair density plot graph with outline
    
    Parameters
    ----------
    i: str
        column name to generate density plot
    
    train_df: df
        training dataframe
        
    Returns
    ----------
    graph: altair object
        density plot with outline
    
    Example
    ----------
    >>> display('Age', train_df)
    '''
    graph = alt.Chart(train_df).transform_density(
    i,groupby=['RiskLevel'],
    as_=[ i, 'density']).mark_area(fill = None, strokeWidth=2).encode(
    x = (i),
    y='density:Q',stroke='RiskLevel').properties(width=200,height=200)
    return graph + graph.mark_area(opacity = 0.3).encode(color = alt.Color('RiskLevel',legend=None))

def boxplot(i,train_df):
    '''
    Return altair boxplot graph
    
    Parameters
    ----------
    i: str
        column name to generate density plot
    
    train_df: df
        training dataframe
        
    Returns
    ----------
    graph: altair object
        boxplot
    
    Example
    ----------
    >>> boxplot('Age', train_df)
    '''
    box = alt.Chart(train_df).mark_boxplot().encode(
    x = i,
    y = 'RiskLevel',
    color = 'RiskLevel')
    return box

def save_chart(chart, filename, scale_factor=1):
    '''
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    '''
    if filename.split('.')[-1] == 'svg':
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split('.')[-1] == 'png':
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")

def main(data_location, output_location):
    '''
    Output EDA figures
    
    Parameters
    ----------
    data_location: string
        location that the data are stored
    
    output_location: string
        location for the EDA figures to be outputted
        
    '''
    
    #read data
    maternal_risk_df = pd.read_csv(data_location)

    train_df, test_df = train_test_split(maternal_risk_df, test_size=0.2, random_state=123) 
    
    corr_df = train_df.corr('spearman').style.background_gradient()
    
    class_distribution = alt.Chart(train_df).mark_bar().encode(
        x = 'count()',
        y = 'RiskLevel',
        color = 'RiskLevel'
    ).properties(title = 'Distribution of Risk Level')

    Age = display('Age',train_df)
    SystolicBP = display('SystolicBP',train_df)
    DiastolicBP = display('DiastolicBP',train_df)
    BS = display('BS',train_df)
    BodyTemp = display('BodyTemp',train_df)
    HeartRate = display('HeartRate',train_df)

    X_density = ((Age | SystolicBP | DiastolicBP) & (BS | BodyTemp | HeartRate)).properties(title='Distribution of Predictors for Each Risk Level')
    
    Age = boxplot('Age',train_df)
    SystolicBP = boxplot('SystolicBP',train_df)
    DiastolicBP = boxplot('DiastolicBP',train_df)
    BS = boxplot('BS',train_df)
    BodyTemp = boxplot('BodyTemp',train_df)
    HeartRate = boxplot('HeartRate',train_df)
    
    X_box = (Age & SystolicBP & DiastolicBP & BS & BodyTemp & HeartRate).properties(title='Boxplots of Different Features')

    combined = (class_distribution & X_density & X_box).configure_title(
        fontSize=18, anchor='middle')

    X_corr = alt.Chart(train_df).mark_point(opacity=0.3, size=10).encode(
        alt.X(alt.repeat('row'), type='quantitative'),
        alt.Y(alt.repeat('column'), type='quantitative')
        ).properties(
            width=100,
            height=100
        ).repeat(
            column=['Age', 'SystolicBP', 'DiastolicBP'],
            row=['Age', 'SystolicBP', 'DiastolicBP'])
            
    try: 
        save_chart(combined, output_location+'EDA.png',1)
    except:
        os.makedirs(os.path.dirname(output_location+'EDA.png'))
        save_chart(combined, output_location+'EDA.png',1)
    
    dfi.export(corr_df, output_location + 'corr_plot.png')
    save_chart(X_density, output_location+'density_plot.png',1)
    save_chart(X_box, output_location+'box_plot.png',1)
    save_chart(class_distribution, output_location+'class_distribution.png',1)
    save_chart(X_corr, output_location+'corr_bp_plot.png',1)
        
    assert os.path.isfile(output_location+'EDA.png'), "EDA is not in the src/maternal_risk_eda_figures directory."
    
if __name__ == "__main__":
  main(opt["--data_location"], opt["--output_location"])

#python src/eda_script.py --data_location='data/raw/maternal_risk.csv' --output_location='src/maternal_risk_eda_figures/'

#save_chart function reference from Joel Ostblom