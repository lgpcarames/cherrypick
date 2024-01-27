__author__ = "Lucas Carames"
__license__="MIT"
__version__='0.1.0'
__maintainer__='Lucas Carames'
__email__='lgpcarames@gmail.com'


import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from tqdm import tqdm
import shap
import warnings
from sklearn.model_selection import StratifiedKFold
from functools import reduce
from typing import Union, Any, List


from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans


warnings.filterwarnings('ignore')
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
        self.dic = {}
        self.rev_dic = {}

    def fit(self, vet: Union[pd.Series, list]) -> 'CatEncoder':
        """
        Prepares the encoder for the data passed in the 'vet' variable.
        """
        uniques = sorted(set(str(c) for c in vet.unique() if isinstance(c, str)))
        self.dic = {b: a for a, b in enumerate(uniques)}
        self.rev_dic = {a: b for b, a in self.dic.items()}
        return self

    def check(self, vet: Union[pd.Series, list]) -> pd.Series:
        """
        Checks the type of the 'vet' variable and converts it to a pd.Series if it is a list.
        """
        if isinstance(vet, list):
            return pd.Series(vet)
        return vet

    def transform(self, vet: pd.DataFrame) -> pd.Series:
        """
        Transforms the label according to the adaptation made by the encoder.
        """
        vet = self.check(vet)
        return vet.map(self.dic)

    def inverse_transform(self, vet: pd.Series) -> pd.Series:
        """
        Reverses the transformation performed by the encoder.
        """
        vet = self.check(vet)
        return vet.map(self.rev_dic)

    def fit_transform(self, vet: pd.Series) -> pd.Series:
        """
        Adapts the encoder to the data and then transforms it.
        """
        vet = self.check(vet)
        self.fit(vet.astype(str))
        return self.transform(vet.astype(str))
    

class SearchHyperParams:
    """
    Aggregates the functions to be used in the study of optimal hyperparameters of the models
    that will be used in the variable selection step.
    """

    def __init__(self, cross_val=5):
        self.lgbm_model = None
        self.lr_model = None
        self.decision_tree_model = None
        self.cross_val = cross_val

    def objective_lr(self, trial: Any, var_train: Union[list, pd.Series, pd.DataFrame],
                     var_test: Union[list, pd.Series, pd.DataFrame]) -> tuple:
        param_grid = {
            "tol": trial.suggest_float("tol", 0.000001, 1.0),
            "max_iter": trial.suggest_int("max_iter", 10, 200),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.00001, 1.0)
        }

        cv_scores = []
        cv_scores_train = []

        cv = StratifiedKFold(n_splits=self.cross_val, shuffle=True, random_state=13)

        for train_idx, test_idx in cv.split(var_train, var_test):
            X_train, X_test = var_train.iloc[train_idx], var_train.iloc[test_idx]
            y_train, y_test = var_test.iloc[train_idx], var_test.iloc[test_idx]

            model = LogisticRegression(class_weight='balanced', penalty='elasticnet', solver='saga',
                                       random_state=13, **param_grid)

            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)

            try:
                cv_scores.append(roc_auc_score(y_test, preds[:, 1]))
                cv_scores_train.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]) - cv_scores[-1])
            except Exception:
                cv_scores.append(-1)
                cv_scores_train.append(1)

        return np.mean(cv_scores), np.mean(cv_scores_train)

    def objective_decision_tree(self, trial: Any, var_train: Union[list, pd.Series, pd.DataFrame],
                                var_test: Union[list, pd.Series, pd.DataFrame]) -> tuple:
        param_grid = {
            "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy']),
            "splitter": trial.suggest_categorical("splitter", ['random', 'best']),
            "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 5, 40),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0001, 0.05),
            'max_depth': trial.suggest_int('max_depth', 2, 50)
        }

        cv_scores = []
        cv_scores_train = []

        cv = StratifiedKFold(n_splits=self.cross_val, shuffle=True, random_state=42)

        for train_idx, test_idx in cv.split(var_train, var_test):
            X_train, X_test = var_train.iloc[train_idx], var_train.iloc[test_idx]
            y_train, y_test = var_test.iloc[train_idx], var_test.iloc[test_idx]

            model = DecisionTreeClassifier(**param_grid)

            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)

            cv_scores.append(roc_auc_score(y_test, preds[:, 1]))
            cv_scores_train.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]) - cv_scores[-1])

        return np.mean(cv_scores), np.mean(cv_scores_train)

    def objective_lgb(self, trial: Any, var_train: Union[list, pd.Series, pd.DataFrame],
                      var_test: Union[list, pd.Series, pd.DataFrame]) -> tuple:
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves": trial.suggest_categorical("num_leaves", [2, 5, 10, 31]),
            "max_depth": trial.suggest_categorical("max_depth", [2, 3, 5]),
            "reg_alpha": trial.suggest_float("lambda_l1", 0, 1),
            "reg_lambda": trial.suggest_float("lambda_l2", 0, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1)
        }

        cv_scores = []
        cv_scores_train = []

        cv = StratifiedKFold(n_splits=self.cross_val, shuffle=True, random_state=42)

        for train_idx, test_idx in cv.split(var_train, var_test):
            X_train, X_test = var_train.iloc[train_idx], var_train.iloc[test_idx]
            y_train, y_test = var_test.iloc[train_idx], var_test.iloc[test_idx]

            model = lgb.LGBMClassifier(objective="binary", **param_grid)

            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc",
                      early_stopping_rounds=100, verbose=False)

            preds = model.predict_proba(X_test)

            cv_scores.append(roc_auc_score(y_test, preds[:, 1]))
            cv_scores_train.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]) - cv_scores[-1])

        return np.mean(cv_scores), np.mean(cv_scores_train)


class CherryPick:
    """
    CherryPick class for feature importance analysis and scoring.

    Parameters:
    -----------
    data: pd.DataFrame
        Input dataframe containing the data.

    variables: list
        List of variable names.

    target: str
        Target variable name.

    lgbModel: object, optional
        Trained LightGBM model.

    treeModel: object, optional
        Trained Decision Tree model.

    data_most_important: pd.DataFrame, optional
        Cached dataframe containing the most important features.

    Methods:
    --------
    gera_shap_score(data, variables, target):
        Generate SHAP scores for feature importance.

    _run_lgbm(data, variables, target):
        Run LightGBM model for feature importance.

    run_tree(data, variables, target):
        Run Decision Tree model for feature importance.

    data_shap_score():
        Classify feature importance according to SHAP scores.

    data_lgbm_gain():
        Classify feature importance according to gain of entropy in a LightGBM model.

    data_lgbm_split():
        Classify feature importance according to split of entropy in a LightGBM model.

    data_logistic_roc():
        Classify feature importance according to the ROC of a logistic model.

    data_mutual_info():
        Classify feature importance according to mutual information.

    data_tree_gain():
        Classify feature importance according to gain of entropy in a Decision Tree model.

    get_feature_importances(logistic_roc=True, mutual_info=True, shap_score=True,
                            tree_gain=True, boost_gain=True, boost_split=True, cache=True):
        Generate a table compiling the results of each metric for feature importance.

    __standard_score__(df, metrics_column):
        Calculate the standard score for ranking metrics.

    __cluster_score__(df, metrics_column, n_cluster=8, init='k-means++', n_init=10,
                      tol=1e-4, verbose=0, random_state=13, algorithm='lloyd'):
        Calculate the cluster score for ranking metrics.

    calculate_score(df, metrics_column, strategy='standard'):
        Calculate the score for ranking metrics based on the specified strategy.
    """

    def __init__(
        self,
        data,
        variables,
        target,
        study_cross_val=5,
        num_studies=50,
        overfit_thres=0.01,
        log_lr_study=False,
        log_lgb_study=False,
        log_tree_study=False,
        verbosity=False,
        baseline=True
    ):
        self.variables = self.__set_variables(variables, baseline)
        if isinstance(target, str):
            self.target = target.replace(' ', '_')
        else:
            raise TypeError('Please check the type of the variable. It must be a string.')
        
        self.data = data

        # Inserting a column with a random variable for comparison
        if 'random_variable' not in self.data.columns.tolist():
            self.data['random_variable'] = np.random.random(size=len(self.data))

        self.study_cross_val = study_cross_val
        self.num_studies = num_studies
        self.overfit_thres = overfit_thres
        self.verbosity = verbosity

        self.log_lr_study = log_lr_study
        self.data_log_lr = None

        self.log_lgb_study = log_lgb_study
        self.data_log_lgb = None

        self.log_tree_study = log_tree_study
        self.data_log_tree = None

        # Output data
        self.data_most_important = None

        # Loading models
        self.lgbModel = None
        self.lrModel = None
        self.treeModel = None
        self.dummyModel = None

        # Loading the objective function
        self.searchParams = SearchHyperParams(cross_val=self.study_cross_val)

        # Redefining the feature names in the inserted data
        self.data.columns = [x.replace(' ', '_') for x in self.data.columns.tolist()]

    def __set_variables(self, var, baseline):
        # Input data
        temp_var = var.copy()
        if baseline:
            temp_var = np.append(temp_var, 'random_variable')
        
        if isinstance(temp_var, list) or isinstance(temp_var, np.ndarray):
            temp_var = [x.replace(' ', '_') for x in temp_var]
        elif isinstance(temp_var, str):
            temp_var = temp_var.replace(' ', '_')
        else:
            raise TypeError('Please check the type of the variable. It must be either a string or a list.')
        return temp_var
    def mutual_info(self, data: pd.DataFrame, explicable_vars: Union[str, list], target: str) -> list:
        """
        Function to calculate the mutual info between the explicable and target variables.

        Parameters:
        ----------
        data: pd.DataFrame
            DataFrame that will be used to choose the most important features
        
        """
        if isinstance(explicable_vars, str):
            df_ = data.dropna(subset=[explicable_vars, target], how='any', axis=0)[[explicable_vars, target]].copy()
        elif isinstance(explicable_vars, list):
            df_ = data.dropna(subset=explicable_vars + [target], how='any', axis=0)[explicable_vars + [target]].copy()
        df_ = df_.reset_index(drop=True)

        return mutual_info_classif(df_.drop(columns=[target]), df_[target], random_state=13)

    def logistic_roc(self, data: pd.DataFrame, explicable_vars: Union[str, list], target: str) -> float:
        """
        Function to obtain the ROC of a logistic regression model trained
        on a specified set of variables.
        The model is optimized using optuna.
        """
        if isinstance(explicable_vars, str):
            df_ = data.dropna(subset=target, how='any', axis=0)[[explicable_vars] + [target]].copy()
        elif isinstance(explicable_vars, list):
            df_ = data.dropna(subset=[target], how='any', axis=0)[explicable_vars + [target]].copy()
        else:
            raise ValueError("Variables must be an instance of list or str")

        df_ = df_.reset_index(drop=True)

        try:
            study = optuna.create_study(directions=['maximize', 'minimize'])
            func = lambda trial: self.searchParams.objective_lr(trial, df_.drop(columns=target).fillna(0), df_[target])
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(func, n_trials=self.num_studies)

            gdsOptuna = study.trials_dataframe()

            if self.log_lr_study:
                self.data_log_lr = gdsOptuna[(gdsOptuna['values_1'] < self.overfit_thres)].sort_values(['values_0', 'values_1'], ascending=[False, False])
            else:
                pass

            dicts = gdsOptuna[(gdsOptuna['values_1'] < self.overfit_thres)].sort_values(['values_0', 'values_1'], ascending=[False, False]).head(10).to_dict(orient='records')

            return dicts[0]['values_0']
        except Exception as e:
            print(e)
            return -1
    def run_tree(self, dataframe: pd.DataFrame, colunas: list, alvo: str) -> DecisionTreeClassifier:
        """
        Runs the decision tree model with the best hyperparameters found through Optuna.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            DataFrame containing the data.
        colunas : list
            List of column names used as features.
        alvo : str
            Target variable name.

        Returns:
        -------
        DecisionTreeClassifier
            Trained decision tree model with the best hyperparameters.
        """
        study = optuna.create_study(directions=['maximize', 'minimize'])
        func = lambda trial: self.searchParams.objective_decision_tree(trial, dataframe[colunas], dataframe[alvo])

        study.optimize(func, n_trials=self.num_studies)

        gdsOptuna = study.trials_dataframe()

        if self.log_tree_study:
            self.data_log_tree = gdsOptuna.sort_values(['values_0', 'values_1'], ascending=[False, False])
        else:
            pass

        dicts = gdsOptuna[(gdsOptuna['values_1'] < self.overfit_thres)].sort_values(['values_0', 'values_1'], ascending=[False, False]).head(10).to_dict(orient='records')
        params = {
            'criterion': dicts[0]['params_criterion'],
            'splitter': dicts[0]['params_splitter'],
            'min_samples_leaf': dicts[0]['params_min_samples_leaf'],
            'min_samples_split': dicts[0]['params_min_samples_split'],
            'max_leaf_nodes': dicts[0]['params_max_leaf_nodes'],
            'ccp_alpha': dicts[0]['params_ccp_alpha'],
            'max_depth': dicts[0]['params_max_depth'],
        }

        self.treeModel = DecisionTreeClassifier(**params)
        self.treeModel.fit(dataframe[colunas], dataframe[alvo])

        return self.treeModel

    def _run_lgbm(self, dataframe: pd.DataFrame, colunas: list, alvo: str) -> lgb.LGBMClassifier:
        """
        Runs the LightGBM model with the best hyperparameters found through Optuna.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            DataFrame containing the data.
        colunas : list
            List of column names used as features.
        alvo : str
            Target variable name.

        Returns:
        -------
        lgb.LGBMClassifier
            Trained LightGBM model with the best hyperparameters.
        """
        study = optuna.create_study(directions=['maximize', 'minimize'])
        func = lambda trial: self.searchParams.objective_lgb(trial, dataframe[colunas], dataframe[alvo])

        study.optimize(func, n_trials=self.num_studies)

        gdsOptuna = study.trials_dataframe()

        if self.log_lgb_study:
            self.data_log_lgb = gdsOptuna.sort_values(['values_0', 'values_1'], ascending=[False, False])
        else:
            pass

        dicts = gdsOptuna[(gdsOptuna['values_1'] < self.overfit_thres)].sort_values(['values_0', 'values_1'], ascending=[False, False]).head(10).to_dict(orient='records')
        params = {
            'colsample_bytree': dicts[0]['params_colsample_bytree'],
            'learning_rate': dicts[0]['params_learning_rate'],
            'max_depth': dicts[0]['params_max_depth'],
            'n_estimators': dicts[0]['params_n_estimators'],
            'num_leaves': dicts[0]['params_num_leaves'],
            'reg_alpha': dicts[0]['params_lambda_l1'],
            'reg_lambda': dicts[0]['params_lambda_l2'],
        }

        self.lgbModel = lgb.LGBMClassifier(objective="binary", **params)
        self.lgbModel.fit(dataframe[colunas], dataframe[alvo], eval_metric="auc", verbose=True)

        return self.lgbModel
    



    def gera_shap_score(self, dataframe: pd.DataFrame, colunas: Union[str, list], alvo: str) -> List:
        """
        Generates SHAP scores and feature importance using LightGBM model.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            DataFrame containing the data.
        colunas : Union[str, list]
            Column names used as features.
        alvo : str
            Target variable name.

        Returns:
        -------
        List
            List containing the SHAP scores DataFrame and the SHAP values.
        """
        if not self.verbosity:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if not dataframe.equals(self.data):
            light_model = self._run_lgbm(dataframe, colunas, alvo)

            explainer = shap.Explainer(light_model, dataframe[colunas])
            shap_values = explainer(dataframe[colunas])

            arr_order = shap_values.abs.mean(0).argsort().flip().values

            optuna.logging.set_verbosity(optuna.logging.DEBUG)

            shap_df = pd.DataFrame(list(zip(np.array(shap_values.feature_names)[arr_order], np.array(shap_values.abs.mean(0).values)[arr_order])), columns=['variable', 'shap_score'])

            return [shap_df, shap_values]

        if not self.lgbModel:
            self._run_lgbm(dataframe, colunas, alvo)

        explainer = shap.Explainer(self.lgbModel, dataframe[colunas])
        shap_values = explainer(dataframe[colunas])

        arr_order = shap_values.abs.mean(0).argsort().flip().values

        optuna.logging.set_verbosity(optuna.logging.DEBUG)

        shap_df = pd.DataFrame(list(zip(np.array(shap_values.feature_names)[arr_order], np.array(shap_values.abs.mean(0).values)[arr_order])), columns=['variable', 'shap_score'])

        return [shap_df, shap_values]


    def data_shap_score(self) -> pd.DataFrame:
        """
        Classify the feature importance according to the SHAP score metric.
        """
        metric = 'shap_score_' + self.target
        print(f'\n--- STARTING SHAP SCORING FOR TARGET: {self.target}---\n')
        aux = self.gera_shap_score(self.data, self.variables, self.target)[0]
        aux = aux.rename(columns={'shap_score': metric})
        print(f'\n--- FINISHED SHAP SCORING FOR TARGET: {self.target}---\n')
        return aux

    def data_lgbm_gain(self) -> pd.DataFrame:
        """
        Classify the feature importance according to the gain of entropy in a boost model.
        """
        print(f'\n--- STARTING BOOST GAIN SCORING FOR TARGET: {self.target}---\n')
        if not self.lgbModel:
            self._run_lgbm(self.data, self.variables, self.target)
        metric = 'lightGBM_gain_' + self.target
        print(f'\n--- FINISHED BOOST GAIN SCORING FOR TARGET: {self.target}---\n')
        return pd.DataFrame(list(zip(self.lgbModel.feature_name_, self.lgbModel.booster_.feature_importance(importance_type='gain'))), columns=['variable', metric]).sort_values(by=metric, ascending=False)

    def data_lgbm_split(self) -> pd.DataFrame:
        """
        Classify the feature importance according to the split of entropy in a boost model.
        """
        print(f'\n--- STARTING BOOST SPLIT SCORING FOR TARGET: {self.target}---\n')
        if not self.lgbModel:
            self._run_lgbm(self.data, self.variables, self.target)
        metric = 'lightGBM_split_' + self.target
        print(f'\n--- FINISHED BOOST SPLIT SCORING FOR TARGET: {self.target}---\n')
        return pd.DataFrame(list(zip(self.lgbModel.feature_name_, self.lgbModel.booster_.feature_importance(importance_type='split'))), columns=['variable', metric]).sort_values(by=metric, ascending=False)

    def data_logistic_roc(self) -> pd.DataFrame:
        """
        Classify the feature importance according to the ROC of a logistic model.
        """
        print(f'\n--- STARTING LOGISTIC ROC SCORING FOR TARGET: {self.target}---\n')
        metric = 'logistic_roc_' + self.target
        dic_ = {
            'variable': [],
            metric: []
        }
        for variable in tqdm(self.variables, desc='logistic-roc metric'):
            dic_['variable'].append(variable)
            dic_[metric].append(self.logistic_roc(self.data, variable, self.target))
        print(f'\n--- FINISHED LOGISTIC ROC SCORING FOR TARGET: {self.target}---\n')
        return pd.DataFrame(dic_).sort_values(by=metric, ascending=False)

    def data_mutual_info(self) -> pd.DataFrame:
        """
        Classify the feature importance according to the mutual information.
        """
        print(f'\n--- STARTING MUTUAL INFORMATION SCORING FOR TARGET: {self.target}---\n')
        metric = 'mutual_info_' + self.target
        dic_ = {
            'variable': [],
            metric: []
        }
        for variable in tqdm(self.variables, desc='mutual information metric'):
            dic_['variable'].append(variable)
            dic_[metric].append(self.mutual_info(self.data, variable, self.target)[0])
        print(f'\n--- FINISHED MUTUAL INFORMATION SCORING FOR TARGET: {self.target}---\n')
        return pd.DataFrame(dic_).sort_values(by=metric, ascending=False)

    def data_tree_gain(self) -> pd.DataFrame:
        print(f'\n--- STARTING DECISION TREE GAIN SCORING FOR TARGET: {self.target}---\n')
        if not self.treeModel:
            self.run_tree(self.data, self.variables, self.target)
        metric = 'decisionTree_gain_' + self.target
        print(f'\n--- FINISHED DECISION TREE GAIN SCORING FOR TARGET: {self.target}---\n')
        return pd.DataFrame(list(zip(self.treeModel.feature_names_in_, self.treeModel.feature_importances_)), columns=['variable', metric]).sort_values(by=metric, ascending=False)


    def get_feature_importances(self, logistic_roc=True, mutual_info=True, shap_score=True,
                                tree_gain=True, boost_gain=True, boost_split=True, cache=True):
        """
        Generate a table compiling the results of each metric for feature importance.

        Parameters:
        -----------
        logistic_roc: bool, optional (default: True)
            If True, include logistic ROC metric in the feature selection pipeline.

        mutual_info: bool, optional (default: True)
            If True, include mutual information metric in the feature selection pipeline.

        shap_score: bool, optional (default: True)
            If True, include SHAP score metric in the feature selection pipeline.

        tree_gain: bool, optional (default: True)
            If True, include tree gain metric in the feature selection pipeline.

        boost_gain: bool, optional (default: True)
            If True, include LGBM gain metric in the feature selection pipeline.

        boost_split: bool, optional (default: True)
            If True, include LGBM split metric in the feature selection pipeline.

        cache: bool, optional (default: True)
            If True, cache the results for future use.

        Returns:
        --------
        pd.DataFrame
            A dataframe containing the compiled feature importances.
        """
        if cache and self.data_most_important is not None:
            return self.data_most_important

        temp_ = []

        if logistic_roc:
            try:
                temp_.append(self.data_logistic_roc())
            except Exception as e:
                print(e)

        if mutual_info:
            try:
                temp_.append(self.data_mutual_info())
            except Exception as e:
                print(e)

        if tree_gain:
            try:
                temp_.append(self.data_tree_gain())
            except Exception as e:
                print(e)

        if shap_score:
            try:
                temp_.append(self.data_shap_score())
            except Exception as e:
                print(e)

        if boost_gain:
            try:
                temp_.append(self.data_lgbm_gain())
            except Exception as e:
                print(e)

        if boost_split:
            try:
                temp_.append(self.data_lgbm_split())
            except Exception as e:
                print(e)

        try:
            self.data_most_important = reduce(lambda left, right: pd.merge(left, right, on='variable'), temp_)
        except:
            print('Something went wrong!')

        return self.data_most_important

    def __standard_score__(self, df, metrics_column):
        """
        Calculate the standard score for ranking metrics.
        """
        df_ = df.sort_values(by=metrics_column[0], ascending=False)
        df_['standard_score'] = range(len(df)-1, -1, -1)

        for metric in metrics_column[1:]:
            df_ = df_.sort_values(by=metric, ascending=False)
            df_['temp_score'] = range(len(df)-1, -1, -1)

            df_['standard_score'] = df_[['standard_score', 'temp_score']].sum(axis=1)

            df_ = df_.drop(columns='temp_score')

        return df_.sort_values(by='standard_score', ascending=False)

    def __cluster_score__(self, df, metrics_column, n_cluster=8, init='k-means++', n_init=10,
                          tol=1e-4, verbose=0, random_state=13):
        """
        Calculate the cluster score for ranking metrics.
        """
        df_ = df.sort_values(by=metrics_column[0], ascending=False)

        df_scaled = df_.copy()

        for col in metrics_column[1:]:
            scaler = MinMaxScaler()
            df_scaled[col] = scaler.fit_transform(df_[[col]])

        cluster = KMeans(n_clusters=n_cluster,
                        init=init,
                        n_init=n_init,
                        tol=tol,
                        verbose=verbose,
                        random_state=random_state,
                        )
        cluster.fit(df_scaled[metrics_column[1:]])

        df_['clusters'] = cluster.predict(df_scaled[metrics_column[1:]])
        df_['cluster_score'] = df_scaled[metrics_column[1:]].mean(axis=1)

        return df_.sort_values(by='cluster_score', ascending=False)

    def competitive_score(self, df: pd.DataFrame, metrics_column: list, strategy='standard'):
        """
        Given the dataframe, and the columns with the metrics. Calculates the score by ranking the metrics.
        The first column of the dataframe must be the variables column. All others must be numeric.

        Parameters:
        -----------
        df: pd.DataFrame
            Dataframe containing the metrics.


        metrics_column: list
            List of columns containing the metrics.

        strategy: str, optional (default: 'standard')
            The scoring strategy to use. Can be 'standard' or 'cluster'.

        Returns:
        --------
        pd.DataFrame
            Dataframe sorted by the calculated score.
        """
        if strategy == 'standard':
            return self.__standard_score__(df=df, metrics_column=metrics_column)
        elif strategy == 'cluster':
            return self.__cluster_score__(df=df, metrics_column=metrics_column)




def threshold_score(predictions: Union[list, np.ndarray], target: Union[list, np.ndarray]) -> dict:
    """
    Calculate the optimal classification threshold based on predicted probabilities and target values.

    Parameters:
    -----------
    predictions : Union[list, np.ndarray]
        Iterable with predicted probabilities from the target variable.

    target : Union[list, np.ndarray]
        Iterable with the actual information from the target variable.

    Returns:
    --------
    threshold_score : dict
        Dictionary with information on the optimal threshold and ranking metrics such as precision, recall,
        accuracy, ROC-AUC, and F1-score.

    Developed by: Vinícius Ormenesse - https://github.com/ormenesse
    """
    fpr, tpr, threshold = roc_curve(target, predictions)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]

    tn, fp, fn, tp = confusion_matrix(target, [1 if item >= list(roc_t['threshold'])[0] else 0 for item in predictions]).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    f_score = (2 * precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f-score': f_score,
        'roc-auc': roc_auc_score(target, predictions),
        'threshold': float(roc_t['threshold'])
    }


def __get_features_threshold_score__(df: pd.DataFrame, variables: Union[list, np.ndarray], target: str) -> pd.DataFrame:
    """
    Obtain the optimal threshold classification for each specified variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe that contains the data for the explanatory variable and the target variable.

    variables : Union[list, np.ndarray]
        List with the names of the explanatory variables for which the optimal classification
        threshold is to be obtained.

    target : str
        Name of the target variable.

    Returns:
    --------
    __get_features_threshold_score__ : pd.DataFrame
        Dataframe containing, for each explanatory variable, its normalized and non-normalized threshold.
        The threshold specifies from which value we can classify the target variable to optimize specificity
        and sensitivity.

    """
    temp = df[[target]]

    array_thres = []
    for variable in variables:
        scale = MinMaxScaler()
        temp[variable] = scale.fit_transform(df[[variable]])
        temp[variable] = temp[variable].fillna(-1)

        limiar_temp = threshold_score(temp[variable], temp[target])
        limiar_temp.update({'variable': variable})
        limiar_temp.update({'threshold_variable': scale.inverse_transform([[limiar_temp['threshold']]])[0][0]})

        array_thres.append(pd.DataFrame(limiar_temp, index=[0]))

    return pd.concat(array_thres, axis=0)


def __best_threshold_classification__(df: pd.DataFrame, variables: Union[list, np.ndarray], target: str) -> pd.DataFrame:
    """
    Perform the prediction of the target variable based on each of its explanatory variables.
    The classification is carried out separately for each explanatory variable, using only their
    respective optimal threshold values. The threshold values are sorted based on the highest
    roc rating.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe that contains the data for the explanatory variables and the target variable.

    variables : Union[list, np.ndarray]
        List with the names of the explanatory variables.

    target : str
        Name of the target variable.

    Returns:
    --------
    __best_threshold_classification__ : pd.DataFrame
        A dataframe where each column represents the classification based on the optimal threshold
        for each variable.

    """
    temp_ = __get_features_threshold_score__(df, variables, target)

    df_temp = df.copy()

    for variable in variables:
        temp_thres = temp_[temp_['variable'] == variable]['threshold_variable'][0]

        # Classify based on the optimal threshold to maximize roc-auc
        temp_list_geq_thres = df_temp[variable].apply(lambda x: 1 if x >= temp_thres else 0)
        score_geq_thres = roc_auc_score(df_temp[target], temp_list_geq_thres)

        temp_list_leq_thres = df_temp[variable].apply(lambda x: 1 if x <= temp_thres else 0)
        score_leq_thres = roc_auc_score(df_temp[target], temp_list_leq_thres)

        if score_geq_thres >= score_leq_thres:
            df_temp[variable] = temp_list_geq_thres
        else:
            df_temp[variable] = temp_list_leq_thres

    return df_temp




def __set_difficulty_group__(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Define the difficulty groups for classifying the rows based on the accuracy rate of the explanatory variables.
    Rows with an accuracy rate higher than the average will be classified as easy, while rows with an accuracy rate
    lower than the average will be classified as difficult.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe that contains the data for the explanatory variables and the target variable.

    target : str
        Name of the target variable.

    Returns:
    --------
    __set_difficulty_group__ : pd.DataFrame
        Dataframe with an additional column indicating the difficulty group of each row.

    Raises:
    -------
    Exception
        If the target variable is not binary.

    """

    rate_0 = (df.drop(columns=target) == 0).sum(axis=1) / df.drop(columns=target).shape[1]
    rate_1 = (df.drop(columns=target) == 1).sum(axis=1) / df.drop(columns=target).shape[1]

    # Using numpy.where to compute success_list based on the target
    success_list = np.where(df[target] == 0, rate_0, rate_1)

    # Convert the result to a list
    success_list = success_list.tolist()


    df['success_rate'] = success_list

    # difficulty_threshold = df['success_rate'].mean()

    # # Set the difficulty group: 0 for easy lines, 1 for difficult lines
    # df['difficulty_group'] = df['success_rate'].apply(lambda x: 0 if x >= difficulty_threshold else 1)

    
    difficulty_threshold_g1 = df['success_rate'].quantile(0.333)

    difficulty_threshold_g2 = df['success_rate'].quantile(0.666)

    # Set the difficulty group: 0 for easy lines, 1 for difficult lines
    df['difficulty_group'] = df['success_rate'].apply(
                        lambda x: 0 if x <= difficulty_threshold_g1 else (1 if x <= difficulty_threshold_g2 else 2)
                        )

    # df['difficulty_group'] = df['success_rate'].apply(lambda x: 0 if x >= difficulty_threshold else 1)

    return df






def __generate_stats_success__(
                            df: pd.DataFrame,
                            variables: Union[list, np.ndarray],
                            target: str,
                            g0_weight=[0.1, 0.3, 1.0],
                            g1_weight=[0.25, 0.5, 1.0],
                            g2_weight=[0.35, 0.65, 1.0]
                            ) -> pd.DataFrame:
    """
    Generate the cherry score along with the statistics used to calculate it.
    The statistics include the success rate range of each variable between the groups of greater and lesser difficulty
    in classification, quartile of the variable in relation to the groups, and the cherry score.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe that contains data referring to the explanatory variable and the target variable.

    variables : Union[list, np.ndarray]
        Iterable with the variables for which the cherry score is to be calculated.

    target : str
        Name of the target variable.

    g0_weight : list
        Weight given according to the success rate of a variable in the group of easiest lines.
        The weight is associated with the quartile that the variable assumes in the group.
        The lower the quartile, the fewer number of hits the variable had in the group,
        therefore, the weight will be lower.

    g1_weight : list
        Weight given according to the success rate of a variable in the group of the lines
        with intermediate difficulty. The weight is associated with the quartile that the
        variable assumes in the group. The lower the quartile, the fewer number of hits the
        variable had in the group, therefore, the weight will be lower.

    g2_weight : list
        Weight given according to the success rate of a variable in the group of the most difficult lines.
        The weight is associated with the quartile that the variable assumes in the group.
        The lower the quartile, the fewer number of hits the variable had in the group,
        therefore, the weight will be lower.
    Returns:
    --------
    __generate_stats_success__ : pd.DataFrame
        Dataframe with the cherry score, along with the statistical metrics used for its calculation.

    """

    df_ = pd.DataFrame({'variable': variables})

    # Vectorized operation for right classification
    right_classification = df[variables].eq(df[target], axis=0)

    # Add difficulty group information
    right_classification['difficulty_group'] = df['difficulty_group']

    # print(right_classification)
    # print(right_classification.groupby('difficulty_group').mean())

    # Compute success rates for each variable in each difficulty group
    success_rate_g0 = right_classification.groupby('difficulty_group').mean().loc[0]
    success_rate_g1 = right_classification.groupby('difficulty_group').mean().loc[1]
    success_rate_g2 = right_classification.groupby('difficulty_group').mean().loc[2]

    # Add success rates to the dataframe
    df_['success_rate_g0'] = success_rate_g0[variables].values
    df_['success_rate_g1'] = success_rate_g1[variables].values
    df_['success_rate_g2'] = success_rate_g2[variables].values

    # Compute quantiles for each group
    quantile_g0 = df_['success_rate_g0'].quantile([0.33, 0.66])
    quantile_g1 = df_['success_rate_g1'].quantile([0.33, 0.66])
    quantile_g2 = df_['success_rate_g2'].quantile([0.33, 0.66])

    # Assign quantile categories
    df_['g0_quantile'] = pd.cut(df_['success_rate_g0'], bins=[-float('inf'), quantile_g0[0.33], quantile_g0[0.66], float('inf')], labels=['Q1', 'Q2', 'Q3'])
    df_['g1_quantile'] = pd.cut(df_['success_rate_g1'], bins=[-float('inf'), quantile_g1[0.33], quantile_g1[0.66], float('inf')], labels=['Q1', 'Q2', 'Q3'])
    df_['g2_quantile'] = pd.cut(df_['success_rate_g2'], bins=[-float('inf'), quantile_g2[0.33], quantile_g2[0.66], float('inf')], labels=['Q1', 'Q2', 'Q3'])

    # Mapping weights to quantiles
    df_['g0_quantile_score'] = df_['g0_quantile'].map({'Q1': g0_weight[0], 'Q2': g0_weight[1], 'Q3': g0_weight[2]}).astype(float)
    df_['g1_quantile_score'] = df_['g1_quantile'].map({'Q1': g1_weight[0], 'Q2': g1_weight[1], 'Q3': g1_weight[2]}).astype(float)
    df_['g2_quantile_score'] = df_['g2_quantile'].map({'Q1': g2_weight[0], 'Q2': g2_weight[1], 'Q3': g2_weight[2]}).astype(float)

    # Compute cherry score
    df_ = df_.assign(cherry_score=lambda x: (
                                            x['g0_quantile_score'] * x['success_rate_g0'] +\
                                                  x['g1_quantile_score'] * x['success_rate_g1'] +
                                                  x['g2_quantile_score']*x['success_rate_g2']) / 3)

    # Sort the DataFrame
    df_ = df_.sort_values(by='cherry_score', ascending=False)


    return df_




def cherry_score(df: pd.DataFrame, variables: Union[list, np.ndarray], target: str, only_score=True):
    """
    Function that sets the pipeline to properly calculate the cherry score.

    Parameters:
    ----------
    df: pd.DataFrame
    Dataframe that contains data referring to the explanatory variable and the target variable.

    variables: Union[list, np.ndarray]
    Iterable with the variables you want to calculate the cherry score

    target: str
    Target variable name

    only_score: bool
    if True, returns a dataframe with features names and cherry_score, if False, also returns all the
    statiscal features that are used to obtain the cherry_score.

    Returns
    -------
    generate_cherry_score: pd.DataFrame
    Dataframe with the cherry score or the entire dataframe with cherry score statistics,
    depending on the value of `only_score`.
    """
    assert not isinstance(variables, str), 'TypeError: variables must not be a string type variable'

    try:
        iter(variables)
    except TypeError:
        raise TypeError('variables must be an iterable')

    if len(variables)==1:
        classified_df = __best_threshold_classification__(df=df, variables=variables, target=target)

        df_difficulty = __set_difficulty_group__(df=classified_df, target=target)

        return pd.DataFrame({'variable': variables, 'cherry_score': df_difficulty['success_rate'].sum()/df_difficulty.shape[0]})


    classified_df = __best_threshold_classification__(df=df, variables=variables, target=target)

    # Creating a column with difficulty group
    df_difficulty = __set_difficulty_group__(df=classified_df, target=target)

    df_score = __generate_stats_success__(df=df_difficulty, variables=variables, target=target)

    if only_score:
        return df_score[['variable', 'cherry_score']]
    else:
        return df_score
      
   


if __name__ == '__main__':
    pass
