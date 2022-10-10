# Santander Customer Satisfaction
# Funções auxiliares // Auxiliary functions


# Imports
import joblib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import randint as sp_randint
import sklearn
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score

# Seed
np.random.seed(31415)

# Identificação das variáveis  // Variables identification
def identify_variables_type(df, target, n = 10):
    """
    Identifica variáveis categóricas e numéricas com base na quantidade de categorias de cada variável.
    Por padrão, variáveis com 10 ou menos categorias são consideradas categóricas.
    
    Identifies categorical and numerical features based on the quantity of categories.
    By default, features with less than or equal to 10 classes are categorical.
    """
    cat_features = []
    num_features = []
    
    for col in df.columns.drop(target):
        if df[col].nunique() <= n:
            cat_features.append(col)
        else:
            num_features.append(col)
            
    return cat_features, num_features


# Conversão dos tipos de dados das variáveis // Conversion of variables data types
def convert_variables_dtype(df, cat_features, num_features, target):
    """
    Converte o tipo de dados de variáveis categóricas, numéricas e alvo.
    Converts the data type of categorical, numerical, and target variables.
    """
    for feat in cat_features:
        df[feat] = df[feat].astype('category')
        
    for feat in num_features:
        df[feat] = df[feat].astype('float64')
        
    df[target] = df[target].astype('category')
    
    return df


# Funções auxiliares para apurar atributos com valor zero em todos os registros
def is_in(elt, seq):
    return any(x is elt for x in seq)

def is_equal(elt, seq):
    return all(x is elt for x in seq)

def identify_cols_zero(df):
    cols_zero = []
    for col in df.columns:
        if is_equal(0, df[col]):
            cols_zero.append(col)
    return cols_zero
    
# Transformação log das variáveis numéricas para tratamento dos outliers

def transform_log_skewness(df, num_features):
    
    # Cálculo da assimetria
    skewness = {}
    for feat in num_features:
        skewness.update({feat: df[feat].skew()})
    
    # Transformação de log + 1 se a assimetria for positiva
    if skewness[feat] > 1:
        df[feat] = np.log1p(df[feat])
    
    # Transformação exponencial se a assimetria for negativa
    elif skewness[feat] < 1:
        df[feat] = np.exp(df[feat])
    
    return df


# Calculando limites superior e inferior para detecção dos outliers // Calculating upper and lower limits for outlier detection
def remove_outliers(df, num_features):
    """
    Identifica e remove valores extremos cujo valor está a 3 desvios-padrão da média (acima ou abaixo).
    Identifies and removes outliers whose value is 3 standard deviations from the mean (above or below).
    """
    # Quantidade de registros
    num_row = df.shape[0]
    print(f'Número de linhas antes de filtrar valores extremos (outliers): {num_row}\n')
    
    # Valores extremos estão abaixo do limite inferior ou acima do limite superior
    for col in num_features:
        sup_limit = df[col].mean() + 3 * df[col].std()
        inf_limit = df[col].mean() - 3 * df[col].std()

        # Apurando número de registros com outliers para cada coluna
        num_abs_outliers = df[(df[col] <= inf_limit) | (df[col] >= sup_limit)].shape[0]
        num_rel_outliers = round(100 * num_abs_outliers / num_row, 2)
        print(f'Número de outliers em {col}: {num_abs_outliers} -> {num_rel_outliers}%\n')

    # Remoção de outliers
    df_new = df[(df[col] > inf_limit) | (df[col] < sup_limit)]
    return df_new


# Identificação de atributos com correlação superior a 0.9 (indicação de multicolinearidade)
def identify_high_corr_var(df, corr_limit):
    """
    Identifica atributos com correlação superior a 0.9 (indicação de multicolinearidade)
    Identifies attributes with correlation greater than 0.9 (indication of multicollinearity)
    """
    # Matriz com filtro de correlação maior ou igual ao limite informado
    high_corr_df = df[df >= corr_limit]
    correlated_features = []
    
    dict_corr = {}
    for idx in high_corr_df.index:
        for col in high_corr_df.columns:
            corr = high_corr_df.loc[idx, col]
            if (idx != col) & (corr >= corr_limit):
                if corr not in dict_corr.keys():
                    value = (idx, col)
                    value = sorted(tuple(value))
                    dict_corr.update({corr: value})
                    if value[1] not in correlated_features:
                        correlated_features.append(value[1])
    return  dict_corr, correlated_features


def evaluate_classification_model(y_test, y_pred, y_pred_proba):
    """ 
    Avalia modelos de classificação por meio de matriz de confusão, AUC, curva ROC e acurácia.
    Evaluates classification models through confusion matrix, AUC, ROC curve and accuracy.
    """
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Extração de cada valor da Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Cálculo da métrica global AUC (Area Under The Curve) com dados reais e previsões em teste
    roc_auc = roc_auc_score(y_test, y_pred)

    # Cálculo da curva ROC com dados e previsões em teste
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # AUC em teste
    auc_ = auc(fpr, tpr)

    # Acurácia em teste
    accuracy = accuracy_score(y_test, y_pred)
    
    return cm, roc_auc, auc_, accuracy



def feature_importance(model, cols_list):
    """ 
    Imprime as 10 variáveis mais importantes para o resultado do modelo.
    Prints the 10 most important variables for the model result
    """
    indices = np.argsort(-abs(model.coef_))

    print("Top 10 - Variáveis mais importantes para o resultado do modelo:")
    print(50*'-')
    for feature in cols_list[indices][0,:10]:
        print(feature) 



def save_model(model_name, model):
    """
    Salva o modelo em disco com pickle.
    Save the model to disk with pickle.
    """
    with open(f'../models/{model_name}.pkl', 'wb') as pickle_file:
        joblib.dump(model, pickle_file)


# Função para seleção de hiperparâmetros com Random Forest // Function for selecting hyperparameters with Random Forest
def random_forest_param_selection(X_train, y_train):
    """
    Função para seleção de hiperparâmetros com Random Forest. 
    Function for selecting hyperparameters with Random Forest.
    """
    n_estimators = [100, 200, 300, 400, 500]
    max_depth = [3, None]
    max_features = sp_randint(1, 11)
    min_samples_split = sp_randint(2, 11)
    min_samples_leaf = sp_randint(2, 11) 
    bootstrap = [True, False]
    criterion = ["gini", "entropy"]
    n_iter_search = 20
        
    param_dist = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'max_features': max_features,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap,
                  'criterion': criterion
                  }
    rand_search = RandomizedSearchCV(RandomForestClassifier(),
                                     param_distributions = param_dist,
                                     n_iter = n_iter_search,
                                     scoring = 'roc_auc',
                                     n_jobs  = -1)
    rand_search.fit(X_train, y_train)
    rand_search.best_estimator_
    return rand_search.best_estimator_


# Função para seleção de hiperparâmetros com Decision Tree
def decision_tree_param_selection(X_train, y_train):
    """ Função para seleção de hiperparâmetros com Decision Tree. """
    max_features = sp_randint(1, 11)
    criterion = ['entropy', 'gini', 'log_loss']
    min_samples_split= [2, 3, 4, 5, 7]
    min_samples_leaf= [1, 2, 3, 4, 6]
    max_depth= [2, 3, 4, 5, 6, 7]
    n_iter_search = 20
    param_dist = {'max_features': max_features,
                  'criterion': criterion,
                  'min_samples_split':min_samples_split,
                  'min_samples_leaf':min_samples_leaf,
                  'max_depth':max_depth}
    
    rand_search = RandomizedSearchCV(DecisionTreeClassifier(),
                                     param_distributions = param_dist,
                                     n_iter = n_iter_search,
                                     scoring = 'roc_auc',
                                     n_jobs  = -1)
    rand_search.fit(X_train, y_train)
    rand_search.best_estimator_
    return rand_search.best_estimator_
    
    
# Função para seleção de hiperparâmetros com Gradient Boosting

def gradient_boosting_param_selection(X_train, y_train):
    """ Função para seleção de hiperparâmetros com Gradient Boosting Classifier. """
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    n_estimators = [50, 100, 150, 200, 300, 400, 500, 600]
    criterion = ['friedman_mse', 'squared_error']
    min_samples_split= [2, 3, 4, 5, 7]
    min_samples_leaf= [1, 2, 3, 4, 6]
    max_depth= [2, 3, 4, 5, 6, 7]
    n_iter_search = 20
    param_dist = {'learning_rate': learning_rate,
                  'n_estimators': n_estimators,
                  'criterion': criterion,
                  'min_samples_split':min_samples_split,
                  'min_samples_leaf':min_samples_leaf,
                  'max_depth':max_depth}
    
    rand_search = RandomizedSearchCV(GradientBoostingClassifier(),
                                     param_distributions = param_dist,
                                     n_iter = n_iter_search,
                                     scoring = 'roc_auc',
                                     n_jobs  = -1)
    rand_search.fit(X_train, y_train)
    rand_search.best_estimator_
    return rand_search.best_estimator_
 

# Função para seleção de hiperparâmetros com XGBoost
# https://xgboost.readthedocs.io/en/latest/parameter.html

def xgb_param_selection(X_train, y_train, nfolds):
    """ Função para seleção de hiperparâmetros com XGBoost. """
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    max_depth= [2, 3, 4, 5, 6, 7]
    param_grid = {'eta': learning_rate,
                  'max_depth': max_depth}
    
    grid_search = GridSearchCV(XGBClassifier(),
                                     param_grid = param_grid,
                                     cv = nfolds,)

    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    return  grid_search.best_params_
