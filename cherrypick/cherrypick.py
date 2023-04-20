import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from functools import reduce
from typing import Union, Any
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


# Categorical features process
class CatEncoder:
    """
    Similarly to SKLearn LabelEncoder, but it does work with new categories.

    To be use:
    ----------
    ce = CatEncoder()
    ce.fit(dataframe[selected_columns])

    ce.transform(dataframe[selected_columns])
    

    Developed by: Vinícius Ormenesse - https://github.com/ormenesse
    """

    def __init__(self) -> None:
        """
        Construtor da classe
        """
        self.dic = {}
        self.rev_dic = {}

    def fit(self, vet: Union[pd.Series, list]) -> None:
        """
        Prepara o encoder para os dados passados na variável vet.
        """
        uniques = []
        for c in vet.unique():
            if str(type(c)) == "<class 'str'>":
                uniques.append(c)
        uniques.sort()
        for a, b in enumerate(uniques):
            self.dic[b] = a
            self.rev_dic[a] = b
        return self
    def check(
            self,
            vet: Union[ pd.Series, list]
            ) -> pd.Series:
        """
        Checa o tipo da variável vet. Transformando em pd.Series, caso seja do tipo list.
        """
        try:
            if type(vet) == list:
                return pd.Series(vet)
            return vet
        except Exception:
            return vet
    def transform(self, vet: pd.DataFrame) -> pd.Series:
        """
        Realiza a transformação do label de acordo com o que foi adaptado pelo encoder.
        """
        vet = self.check(vet)
        return vet.map(self.dic)
    
    def inverse_transform(self, vet: pd.Series) -> pd.Series:
        """
        Inverte a transformação realizada pelo encoder.
        """
        vet = self.check(vet)
        return vet.map(self.rev_dic)
    
    def fit_transform(self, vet: pd.Series) -> pd.Series:
        """
        Adapta o encoder aos dados e em seguida os transforma.
        """
        vet = self.check(vet)
        self.fit(vet.astype(str))
        return self.transform(vet.astype(str))
    

class SearchHyperParams:
    def __init__(self, dataframe, variables, target):
        # Loading model variables
        self.lgbm_model = None
        self.lr_model = None
        self.decision_tree_model = None

        # Loading dataframe, variables and target
        self.df = dataframe
        self.vars = variables
        self.target = target
        self.cross_val = 5


    def objective_lr(self, trial: Any, var_train, var_test) -> tuple:
        param_grid = {
            "tol": trial.suggest_loguniform("tol", 0.000001, 1.0),
            "max_iter": trial.suggest_int("max_iter", 10, 200),
            "l1_ratio": trial.suggest_loguniform("l1_ratio", 0.00001, 1.0)
        }

        cv = StratifiedKFold(n_splits=self.cross_val, shuffle=True, random_state=13)

        cv_scores = np.empty(self.cross_val)
        cv_scores_train = np.empty(self.cross_val)

        for idx, (train_idx, test_idx) in enumerate(cv.split(var_train, var_test)):
            X_train, X_test = var_train.iloc[train_idx], var_train.iloc[test_idx]
            y_train, y_test = var_test.iloc[train_idx], var_test[test_idx]


            model = LogisticRegression(class_weight = 'balanced',
                                        penalty='elasticnet',
                                        solver='saga',
                                        random_state=13,
                                        **param_grid)

            model.fit(
                X_train,
                y_train
            )

            preds = model.predict_proba(X_test)

            try:
                cv_scores[idx] = roc_auc_score(y_test, preds[:, 1])
                cv_scores_train[idx] = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]) - cv_scores[idx]
            except Exception:
                cv_scores[idx] = -1
                cv_scores_train[idx] = 1
        return np.mean(cv_scores), np.mean(cv_scores_train)


    def objective_decision_tree(self, trial: Any, var_train, var_test):
        """
        Função de estudo do optuna adaptada para Árvore de Decisão.
        Paramêtros
        ----------
        trial: Função processo de avaliação optuna
        x: Variáveis Explicativa
        y: Nome da variável alvo.
        Retorna
        -------
        tuple -> roc-auc média do modelo base de dados de teste, diferença da roc-auc de teste e
        do treinamento. 
        """
        param_grid = {
            "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy']),
            "splitter": trial.suggest_categorical("splitter", ['random', 'best']),
            "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 5, 40),
            'ccp_alpha': trial.suggest_loguniform('ccp_alpha', 0.0001, 0.05),
            'max_depth': trial.suggest_int('max_depth', 2, 50)

            
        }

        cv = StratifiedKFold(n_splits=cvs, shuffle=True, random_state=42)

        cv_scores = np.empty(cvs)
        cv_scores_train = np.empty(cvs)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = tree.DecisionTreeClassifier(
                **param_grid
            )

            model.fit(
                X_train,
                y_train
            )

            preds = model.predict_proba(X_test)
            cv_scores[idx] = roc_auc_score(y_test, preds[:,1])
            cv_scores_train[idx] = roc_auc_score(y_train,model.predict_proba(X_train)[:,1]) - cv_scores[idx]
            
        return np.mean(cv_scores), np.mean(cv_scores_train)
        
        


if __name__ == '__main__':
    pass