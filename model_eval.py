from sklearn.metrics import precision_recall_curve
from numpy import argmax
from sklearn.metrics import f1_score,precision_score,recall_score,plot_confusion_matrix,plot_precision_recall_curve
import seaborn as sns
import pandas as pd
import numpy as np

def evaluate_model(model, X_test, y_test, threshold=0.5):
    
    model.verbose = False
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > threshold).astype('int')
    
    f1=f1_score(y_test, y_pred)
    p=precision_score(y_test, y_pred)
    r=recall_score(y_test, y_pred)
       
    print(plot_precision_recall_curve(model, X_test, y_test))

    print(plot_confusion_matrix(model,
                          X_test,
                          y_test,
                          normalize='true',
                          display_labels=['NO MATCH', 'MATCH']))
    print(f"f1 = {f1} precission = {p} recall = {r}")
    return f1,p,r
    
# two model by prob and by count
def declie_plot(model,X_test,y_test):

    y_proba = model.predict_proba(X_test)
    y_proba = y_proba[:, 1]
    y_pred = model.predict(X_test)
    
    decile_df = pd.DataFrame(y_proba,columns=['predict_proba'])
    decile_df['actual'] = y_test.values
    decile_df['decile_rank'] =pd.cut(decile_df['predict_proba'],10,labels=False)
    decile_df['Predicted'] = y_pred
    plot_df = decile_df.groupby(['decile_rank'])['actual'].value_counts(normalize=True).rename("percentage").mul(100).reset_index()
    
    g = sns.barplot(plot_df['decile_rank'],y=plot_df['percentage'],hue=plot_df['actual'])

    sns.set(rc={'figure.figsize':(6,6)})
    # show numbers as well 
    for index,row in plot_df.iterrows():
        g.text(row.decile_rank,row.percentage,round(row.percentage,2),color='black',ha='center')