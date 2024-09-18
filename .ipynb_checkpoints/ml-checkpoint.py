import pandas as pd
import numpy as np
from matplotlib import pyplot
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score as accuracy
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, chi2, RFE

from scipy.stats import mode

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

from .utils import validate_model

import shap 

import plotly
from plotly import graph_objects as go 


def plot_roc_curve_(y, y_pred):
    # calculate the fpr, tpr, AUC and plot the ROC curve
    '''
    input:
    y: array-like contenente i valori dell'attributo target
    y_pred: array-like contenente le predizioni dell'attributo target
    output: None

    '''
    y = np.array(y)       #cast array-like in numpy array
    y_pred = np.array(y_pred)
    y = MinMaxScaler(feature_range=(0,1)).fit_transform(y.reshape(-1, 1)) 
    y_pred = MinMaxScaler(feature_range=(0,1)).fit_transform(y_pred.reshape(-1, 1))
    labels = np.unique(y)
    fpr, tpr, threshold = roc_curve(y, y_pred)
    #print(fpr, tpr)
    roc_auc = round(auc(fpr, tpr),2)

    plt.title('ROC curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC {}'.format(roc_auc))
    plt.legend(loc = 'lower right')
    plt.plot(labels, labels,'r--')
    plt.xlim(labels)
    plt.ylim(labels)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    
def train_test_split(data, test_size = 0.7, target_name = 'Target'):
    #data: pd.DataFrame
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data.drop(target_name, axis=1).values, data[target_name].values, test_size=test_size, stratify = data[target_name])
    return X_train, X_test, y_train, y_test


def baselineComparison(X, y, scoring = 'accuracy', class_weight=True):
    '''input:
    X: array-like
    y: array-like contenente i valori dell'attributo target
    scoring: string che specifica che metrica utilizzare nella crossvalidation, default 'accuracy'
    class_weight: boolean, True se vogliamo fare una classificazione. default True
    output: models tuple (model_name, model)
    '''
    # Array contenente vari algoritmi di classificazione da testare
    models = []

    if(class_weight):
        cw = 'balanced'
    else:
        cw = None
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('NB', GaussianNB()))

    models.append(('LR', LogisticRegression(solver='lbfgs',max_iter=100,class_weight=cw)))
    models.append(('KNN', KNeighborsClassifier(2, weights='distance')))
    models.append(('DT', DecisionTreeClassifier(class_weight=cw)))
    models.append(('SVM', SVC(gamma='scale', class_weight=cw, probability=True)))
    models.append(('RF', RandomForestClassifier(max_depth=5, n_estimators=20, class_weight=cw)))
    #class_weight='balanced' serve per gestire lo sbilanciamento del dataset dando un peso diverso alle classi

    results = []
    namesModels = []
    best_model = None
    best_nameModel = ''
    best_scores = 0.0

    # Addestramento e valutazione algoritmi
    for nameModel, model in models:

        kfold = StratifiedKFold(n_splits=2, shuffle=True)

        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        
        fig, ax = pyplot.subplots(figsize=(6, 6))

        for fold, (train, test) in enumerate(kfold.split(X, y)):

            #smote_train = SMOTE(sampling_strategy = {0: 15, 1: 15}, random_state=42, k_neighbors = 1)
            #X_train_res,  y_train_res = smote_train.fit_resample(X[train], y[train])
            X_train_res,  y_train_res = X[train], y[train]
            X_test_res,  y_test_res = X[test], y[test] #smote_test.fit_resample(X[test], y[test])

            model.fit(X_train_res, y_train_res)

            viz = RocCurveDisplay.from_estimator(
                                                          model,
                                                          X_test_res,
                                                          y_test_res,
                                                          name='{} ROC fold {}'.format(nameModel,fold),
                                                          alpha=0.8,
                                                          lw=2,
                                                          ax=ax
                                                )

            #print(viz.fpr, viz.tpr)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

            if fold == 3:
              break

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability",
        )
        ax.legend(loc="lower right")
        pyplot.show()
                
        if (scoring=='roc_auc'):
            scores = aucs
        else:
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

        if(np.mean(scores) > np.mean(best_scores)):

            best_scores = scores
            best_nameModel = nameModel

        results.append(scores)
        namesModels.append(nameModel)
        msg = "%s: %f (%f)\n" % (nameModel, np.mean(scores),  np.std(scores))
        print(msg)

    print('Best model {} with {}:{}'.format(best_nameModel, scoring, np.mean(best_scores)))
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(namesModels)
    plt.show()

    return models

def models_roc_curves(models, X_test, y_test):

    fig, ax = plt.subplots(figsize=(6, 6))
    best_score = 0.0
    best_model = None
    colors = ["g", "k", "c", "orange", "b", "purple", "pink"]

    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')  # Diagonal line

    for i, (name, model) in enumerate(models):

        color = colors[i]
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        # Assuming y_true are the true labels and y_pred_prob are the predicted probabilities

        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plotting the ROC curve with precise data points
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} ROC curve (area = {roc_auc:.5f})')
        if roc_auc > best_score:
          best_score = roc_auc
          best_model = model

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    #plt.show()
    return fig, best_model, best_score


def plot_feature_importance(importances, genes):

    # Create a DataFrame to view the gene importances
    feature_importances = pd.DataFrame({
        'Gene': genes,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Assuming `feature_importances` DataFrame is already prepared
    top_genes = feature_importances.head(10)

    fig = plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Gene', data=top_genes, palette='viridis')
    plt.title('Top 10 Most Important Genes')
    plt.xlabel('Importance Score')
    plt.ylabel('Gene')
    return fig 

def explain_model(model, X, genes):
      
    # Fit your model 
    
    try:
        explainer = shap.Explainer(model)
    except:
        masker = shap.maskers.Independent(data = X)
        explainer = shap.Explainer(model.predict, masker = masker)
    shap_values = explainer(X)

    # Convert SHAP values to a DataFrame
    # Use shap_values.values[:, :, 1] for binary classification if you want the positive class
    shap_df = pd.DataFrame(shap_values.values[:, :, 1], columns=genes)

    # Calculate mean absolute SHAP values per feature
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

    # Create DataFrame for plotting
    feature_importances = pd.DataFrame({
        'Gene': mean_abs_shap.index,
        'Mean Absolute SHAP Value': mean_abs_shap.values
    }).head(10)  # Select top 10 genes

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
            x=feature_importances["Mean Absolute SHAP Value"].values,
            y=feature_importances["Gene"].values,
            orientation='h')
    )
    fig.update_layout(title = 'Top 10 Most Important Genes Based on SHAP Values', xaxis_title="Mean Absolute SHAP Value", yaxis_title="Gene")

    return fig 

def shap_summary(model, X, genes):
    
    # Create Tree Explainer object that can calculate shap values
    explainer = shap.TreeExplainer(model)

    # Calculate Shap values
    shap_values = explainer.shap_values(X)

    # Make plot
    fig = shap.summary_plot(shap_values[:, :, 1], X, max_display=10, sort = True, feature_names=genes, show = False)
    return fig 

def shap_force(model, X_test, y_test, genes, index = 0):
    
    explainer = shap.Explainer(model)
    
    y = y_test[index]
    choosen_instance = X_test[index, :]
    shap_values = explainer.shap_values(choosen_instance)[:, 0]
    shap.initjs()
    fig = shap.force_plot(explainer.expected_value[1], shap_values, choosen_instance, feature_names=genes, show = False)
    return fig 

def create_results_df(models, X_test, y_test):
    
    df = pd.DataFrame([], columns = ["Model Name", "Accuracy", "F1", "Sensitivity", "Specificity", "AUC score", "Precision"])
    
    for i, (model_name, model) in enumerate(models.items()):

        print(model_name)
        y_pred = model.predict(X_test)
        metrics, msg = validate_model(y_test, y_pred)
        acc = float(round(metrics['accuracy']*100, 2))
        f1 = float(round(metrics['f1'], 2))
        sensitivity = float(round(metrics['sensitivity']*100, 2))
        specificity = float(round(metrics['specificity']*100, 2))
        auc = float(round(metrics['auc_score'], 4))
        precision = float(round(metrics['precision']*100, 2))
        df.loc[i] = [model_name, acc, f1, sensitivity, specificity, auc, precision]

    df = df.sort_values(by = "AUC score", ascending = False)
    return df