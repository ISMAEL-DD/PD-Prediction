# Importer les libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings
pd.set_option('display.max_columns', 500)
warnings.filterwarnings("ignore")
# Importation des données


def importation(database):
    return pd.read_csv(database, low_memory=False)


loan = importation('LoanData.csv')


def preprocessing(loan_data):
    loan_data = loan_data[
        ["NewCreditCustomer", "Age", "Gender", "Country", "Amount", "Interest", "LoanDuration", "UseOfLoan",
         "Education", "MaritalStatus", "EmploymentStatus", "EmploymentDurationCurrentEmployer", "OccupationArea",
         "HomeOwnershipType", "IncomeTotal", "DefaultDate", "AmountOfPreviousLoansBeforeLoan"]]
# Création de notre variable cible "Default" depuis la variable "DefaultDate"
# Creation of the Default variable (our target variable) that takes True
# as a value if DefaultDate is NA and False otherwise
    loan_data["Default"] = loan_data["DefaultDate"].isnull()
    loan_data["Default"] = loan_data["Default"].astype("str")
# Nous remplaçons True par 0 (le client est sain) et False par 1
    # We then replace True by 0 meaning that the client have not default and False by 1
    loan_data["Default"] = loan_data["Default"].replace("True", 0)
    loan_data["Default"] = loan_data["Default"].replace("False", 1)
# so we divide them by 100
    loan_data["Interest"] = loan_data["Interest"] / 100
# Nous supprimons la variable "WorkExperience car la majorité de ses valeurs sont manquantes"
    loan_data["AmountOfPreviousLoansBeforeLoan"] = loan_data["AmountOfPreviousLoansBeforeLoan"].fillna(0)
# pour la "AmountOfPreviousLoansBeforeLoan", les clients pour lesquelles ses valeurs sont manquantes,
# n'ont pas octroyés de crédits antérieurement, nous remplaçons donc ces valeurs par 0
    loan_data.loc[loan_data["Age"] < 18, "Age"] = loan_data["Age"].quantile(0.25)
# Nous remplaçons les valeurs de la variables "Age" qui sont inférieurs à 0 par le premier quartile de cette variable
# loan_data.loc[(loan_data["IncomeTotal"] <= 100) & (loan_data["EmploymentStatus"] != 1)]["IncomeTotal"].unique()
    loan_data.loc[(loan_data["IncomeTotal"] == 0) & (loan_data["EmploymentStatus"] != 1), "IncomeTotal"] = \
        loan_data["IncomeTotal"].quantile(0.25)
# Si un client a un emploi et que son revenu est 0, nous remplaçons ce dernier par le premier quartile
    categorical_columns = ["EmploymentDurationCurrentEmployer", "HomeOwnershipType", "OccupationArea",
                           "EmploymentStatus", "MaritalStatus", "Education", "Gender"]
    loan_data[categorical_columns] = loan_data[categorical_columns].fillna(loan_data.mode().iloc[0])
# Nous remplaçons les valeurs manquantes de chaque variable qualitative par le mode
    outlier_columns = ["UseOfLoan", "MaritalStatus", "EmploymentStatus", "OccupationArea", "HomeOwnershipType"]
    loan_data[outlier_columns] = loan_data[outlier_columns].replace({-1: 1})
# Nous remplaçons les valeurs aberrantes (-1) des variables citées sur la liste
# outliers_columns par 1 => Erreur de saisie
    loan_data = loan_data.loc[(loan_data["OccupationArea"] > 0) & (loan_data["MaritalStatus"] > 0)
                              & (loan_data["EmploymentStatus"] > 0)]
    loan_data = loan_data.drop(["DefaultDate"], axis=1)
# Nous supprimons les ligne ayant la valeur 0 comme modalité au niveau des variables
# "OccupationArea", "MaritalStatus", "EmploymentStatus".
    return loan_data


loan_data_frame = preprocessing(loan)
loan_data_frame.isna().sum()
# On commence par transformer les modalités des variables qualitatives
# ayant comme type float en des modalités de type entier
# Conversion des variables catégorielles en type categoriel
categorical_variables = ["NewCreditCustomer", "Gender", "Education", "MaritalStatus", "EmploymentStatus",
                         "OccupationArea", "HomeOwnershipType", "Default", "UseOfLoan", "Country",
                         "EmploymentDurationCurrentEmployer"]
for variable in categorical_variables:
    loan_data_frame[variable] = loan_data_frame[variable].astype("category")
# Définir les variables explicatives
predictors = loan_data_frame.drop('Default', axis=1)
predictors.info()
# Définir la variable cible
target = loan_data_frame['Default']
# Diviser les variables qualitatives et quatitatives
numeric = predictors.select_dtypes(include=np.number).columns.tolist()[:-1]
categories = predictors.select_dtypes('category').columns.tolist()
# L'encodage des variables qualitatives et standardisation des variables quantitatives
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore", drop='first')
# Creation du pipeline de l'encodage et du modèle statistique
preprocessor = make_column_transformer((encoder, categories), (StandardScaler(), numeric))
# Creation des données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)


def oversampling_undersampling(training_variables, training_target, over=False):
    """
    Pour équilibrer notre base de données
    """
    lotemp = pd.concat([training_variables, training_target], axis=1)
    defaut = lotemp[lotemp["Default"] == 1]
    nondefaut = lotemp[lotemp["Default"] == 0]
    if over:
        oversampled_default = resample(defaut, replace=True, n_samples=len(nondefaut), random_state=42)
        data1 = nondefaut
        data2 = oversampled_default
    else:
        undersampled_non_default = resample(nondefaut, replace=True, n_samples=len(defaut), random_state=42)
        data1 = defaut
        data2 = undersampled_non_default

    loan_new = pd.concat([data1, data2], axis=0)
    target_new = loan_new["Default"]
    predictors_new = loan_new.drop(columns=["Default"], axis=1)
    X_train = predictors_new
    y_train = target_new
    non_default_train = (y_train.values == 0).sum()
    default_train = (y_train.values == 1).sum()
    return loan_new, X_train, y_train, non_default_train, default_train


oversampling_undersampling(X_train, y_train, over=False)
print(oversampling_undersampling(X_train, y_train, over=False)[3])


def acp_inspection(training_variables, training_target):
    """
    Avoir une idée sur le résultat de PCA: savoir le nombre de variables reduits
    preprocessor: pipeline of encoding categorical variables and standardizing the numerical
    :param training_variables are the predictors training set
    :param training_target are is the response variable training set
    """
    # Create a PCA instance: pca
    pca = PCA()
    # Creer pipeline: pipeline
    pipeline1 = make_pipeline(preprocessor, pca)
    # Fit the pipeline to 'samples'
    pipeline1.fit(training_variables, training_target)
    # Plot les variances expliqués
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.xticks(features)
    plt.show()
    print(features)
    # On a une reduction de varibable à 14 avec la meme accuracy


acp_inspection(X_train, y_train)

'''Pipeline'''
# Build the pipeline
# Set up the pipeline steps: steps
steps = [('one_hot', preprocessor),
         ('reducer', PCA(n_components=16)),
         ('classifier', LogisticRegression())]
pipe = Pipeline(steps)
param_dict = {"reducer__n_components": np.arange(4, 20, 2)}


def pca_tune(pipeline1, parameters, training_variables, training_target, testing_variables, testing_target):
    """
    Elle nous permet de determiner les meilleurs params grace a l'evaluation de la precision
    :param pipeline1 is our pipeline
    :param parameters is the hyperparameter that we want to tune
    :param training_variables are the predictors training set
    :param training_target are is the response variable training set
    :param testing_variables are the predictors testing set
    :param testing_target are is the response variable testing set
    """
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    # Create the GridSearchCV object: gm_cv
    gm_cv = GridSearchCV(pipeline1, parameters, cv=cv, scoring='accuracy')
    # Fit the classifier to the training data
    gm_cv.fit(training_variables, training_target)
    # Compute and print the metrics
    print("Accuracy: {}".format(gm_cv.score(testing_variables, testing_target)))
    print(classification_report(testing_target, gm_cv.predict(testing_variables)))
    print("Tuned pca Alpha: {}".format(gm_cv.best_params_))
    return gm_cv.best_params_


pca_tune(pipe, param_dict, X_train, y_train, X_test, y_test)

'''Pipeline knn'''
# Build the pipeline
# Set up the pipeline steps: steps
steps3 = [('one_hot', preprocessor),
          ('reducer', PCA(n_components=16)),
          ('knn', KNeighborsClassifier())]
pipe3 = Pipeline(steps3)
param_dict1 = {'knn__n_neighbors': np.arange(17, 21, 1)}


def knn_tune(pipeline2, parameters, training_variables, training_target, testing_variables, testing_target):
    """
    Elle nous permet de determiner le meilleur k grace a l'evaluation de la precision (Accuracy)
    :param pipeline2 is our pipeline
    :param parameters is the hyperparameter that we want to tune
    :param training_variables are the predictors training set
    :param training_target are is the response variable training set
    :param testing_variables are the predictors testing set
    :param testing_target are is the response variable testing set
    """
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    # Create the GridSearchCV object: gm_cv
    gm_cv1 = GridSearchCV(pipeline2, parameters, cv=cv, scoring='accuracy')
    gm_cv1.fit(training_variables, training_target)
    # Calculer y_pred a travers X_test
    prediction_target = gm_cv1.predict(testing_variables)
    print(classification_report(testing_target, prediction_target))
    print("Tuned knn k: {}".format(gm_cv1.best_params_))
    return gm_cv1.best_params_


knn_tune(pipe3, param_dict1, X_train, y_train, X_test, y_test)


'''pipeline inputs pour random forests'''
# Number of trees in Random Forest
n_estimators = [200, 500, 600]
# Number of features to consider at each split
max_features = ["auto", "log2"]
# Maximum number of levels in tree
max_depth = [80, 100, 120]
# Minimum number of samples required to split a node
min_samples_split = [5, 10, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [10, 20, 50]
# Method of selecting samples for training each tree
bootstrap = [True, False]
criterion = ["gini"]
# Create the random grid
random_grid = {
    "rf__n_estimators": n_estimators,
    "rf__max_features": max_features,
    "rf__max_depth": max_depth,
    "rf__min_samples_split": min_samples_split,
    "rf__min_samples_leaf": min_samples_leaf,
    "rf__bootstrap": bootstrap,
    "rf__criterion": criterion,
}

steps5 = [('one_hot', preprocessor),
          ('reducer', PCA(n_components=16)),
          ('rf', RandomForestClassifier())]
pipe5 = Pipeline(steps5)


def pipeline_random_forest_tune(pipeline5, params, training_variables, training_target, testing_variables,
                                testing_target):
    """
    Création du pipeline avec le random forest comme classifieur
    :param pipeline5 is our pipeline
    :param params is the hyperparameter that we want to tune
    :param training_variables are the predictors training set
    :param training_target are is the response variable training set
    :param testing_variables are the predictors testing set
    :param testing_target are is the response variable testing set
    """
    Randomized_Random_Forest = RandomizedSearchCV(estimator=pipeline5, param_distributions=params,
                                                  scoring='neg_mean_squared_error', n_iter=100, cv=2, verbose=2,
                                                  random_state=42, n_jobs=-1)

    Randomized_Random_Forest.fit(training_variables, training_target)
    # Calculer y_pred a travers X_test
    prediction_target = Randomized_Random_Forest.predict(testing_variables)
    print(confusion_matrix(testing_target, prediction_target))
    print(classification_report(testing_target, prediction_target))
    print(Randomized_Random_Forest.best_estimator_)
    return Randomized_Random_Forest.best_estimator_


pipeline_random_forest_tune(pipe5, random_grid, X_train, y_train, X_test, y_test)


def pipeline_logreg(training_variables, training_target, testing_variables, testing_target):
    """
    Création du pipeline avec la regression logistique comme classifieur
    :param training_variables are the predictors training set
    :param training_target are is the response variable training set
    :param testing_variables are the predictors testing set
    :param testing_target are is the response variable testing set
    """
    steps1 = [('one_hot', preprocessor),
              ('reducer', PCA(n_components=16)),
              ('classifier', LogisticRegression())]
    pipe1 = Pipeline(steps1)
    pipe1.fit(training_variables, training_target)
    # Calculer y_pred a travers X_test
    prediction_target = pipe1.predict(testing_variables)
    print(confusion_matrix(testing_target, prediction_target))
    print(classification_report(testing_target, prediction_target))
    return pipe1


pipeline_logreg(X_train, y_train, X_test, y_test)


def pipeline_knn(training_variables, training_target, testing_variables, testing_target):
    """
    Création du pipeline avec l'algorithme de knn comme classifieur
    :param training_variables are the predictors training set
    :param training_target are is the response variable training set
    :param testing_variables are the predictors testing set
    :param testing_target are is the response variable testing set
    """
    steps4 = [('one_hot', preprocessor),
              ('reducer', PCA(n_components=16)),
              ('knn', KNeighborsClassifier(n_neighbors=15))]
    pipe4 = Pipeline(steps4)
    pipe4.fit(training_variables, training_target)
    # Calculer y_pred a travers X_test
    prediction_target = pipe4.predict(testing_variables)
    print(confusion_matrix(testing_target, prediction_target))
    print(classification_report(testing_target, prediction_target))
    return pipe4


pipeline_knn(X_train, y_train, X_test, y_test)


def pipeline_rf(training_variables, training_target, testing_variables, testing_target):
    """
    Création du pipeline avec l'algorithme de rf comme classifieur
    :param training_variables are the predictors training set
    :param training_target are is the response variable training set
    :param testing_variables are the predictors testing set
    :param testing_target are is the response variable testing set
    """
    steps6 = [('one_hot', preprocessor),
              ('reducer', PCA(n_components=16)),
              ('rf', RandomForestClassifier(n_estimators=450, max_depth=80, min_samples_split=5,
                                            min_samples_leaf=20, max_features="auto", bootstrap=False))]
    pipe6 = Pipeline(steps6)
    pipe6.fit(training_variables, training_target)
    prediction_target = pipe6.predict(testing_variables)
    print(confusion_matrix(testing_target, prediction_target))
    print(classification_report(testing_target, prediction_target))
    return pipe6


pipeline_rf(X_train, y_train, X_test, y_test)


def pipeline_adaboost(training_variables, training_target, testing_variables, testing_target):
    """
    Création du pipeline avec l'algorithme de Adaboost comme classifieur
    :param training_variables are the predictors training set
    :param training_target are is the response variable training set
    :param testing_variables are the predictors testing set
    :param testing_target are is the response variable testing set
    """
    dt = DecisionTreeClassifier(max_depth=2, random_state=1)
    steps7 = [('one_hot', preprocessor),
              ('reducer', PCA(n_components=16)),
              ('ada', AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1))]
    pipe7 = Pipeline(steps7)
    pipe7.fit(training_variables, training_target)
    prediction_target = pipe7.predict(testing_variables)
    print(confusion_matrix(testing_target, prediction_target))
    print(classification_report(testing_target, prediction_target))
    return pipe7


pipeline_adaboost(X_train, y_train, X_test, y_test)


def pipeline_xgboost(training_variables, training_target, testing_variables, testing_target):
    """
    Création du pipeline avec l'algorithme de Adaboost comme classifieur
    :param training_variables are the predictors training set
    :param training_target are is the response variable training set
    :param testing_variables are the predictors testing set
    :param testing_target are is the response variable testing set
    """
    steps8 = [('one_hot', preprocessor),
              ('reducer', PCA(n_components=16)),
              ('XGB', XGBClassifier(objective='binary:logistic', n_estimators=10, max_depth=8))]
    pipe8 = Pipeline(steps8)
    pipe8.fit(training_variables, training_target)
    prediction_target = pipe8.predict(testing_variables)
    print(confusion_matrix(testing_target, prediction_target))
    print(classification_report(testing_target, prediction_target))
    return pipe8


pipeline_xgboost(X_train, y_train, X_test, y_test)


def evaluation_model(model, testing_variables, testing_target):
    """
    Evaluer la performance du modele via les KPIS affichés
    :param model is the model that we want to assess
    :param testing_variables are the predictors in test
    :param testing_target are is the response variable in test
    """
    prediction_target = model.predict(testing_variables)
    print("la matrice de confusion du ", model, "est: \n", confusion_matrix(testing_target, prediction_target))
    print("Le rapport de classification : \n", classification_report(testing_target, prediction_target))
    print("Le roc_auc_score : \n", roc_auc_score(testing_target, prediction_target))
    print("Le f1_score :\n", f1_score(testing_target, prediction_target))
    print("La courbe de roc : \n")
    specificity, sensitivity, _ = roc_curve(testing_target, prediction_target)
    print("sensitivity is:", sensitivity)
    print("specificity is:", specificity)
    plt.plot(specificity, sensitivity)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


evaluation_model(pipeline_logreg(X_train, y_train, X_test, y_test), X_test, y_test)
evaluation_model(pipeline_knn(X_train, y_train, X_test, y_test), X_test, y_test)
evaluation_model(pipeline_rf(X_train, y_train, X_test, y_test), X_test, y_test)
evaluation_model(pipeline_adaboost(X_train, y_train, X_test, y_test), X_test, y_test)
evaluation_model(pipeline_xgboost(X_train, y_train, X_test, y_test), X_test, y_test)