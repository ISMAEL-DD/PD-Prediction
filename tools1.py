## Importer les libraries
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
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


def evaluation_model(model, training_variables, training_target):
    """
    Evaluer la performance du modele via les KPIS affichés
    :param model is the model that we want to assess
    :param training_variables are the predictors
    :param training_target are is the response variable
    """
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    # Compute 3-fold cross-validation scores: cv_scores
    cv_accuracy = cross_val_score(model, training_variables, training_target, cv=cv, scoring='accuracy')
    print("Average 3-Fold CV accuracy: {}".format(np.mean(cv_accuracy)))
    cv_recall = cross_val_score(model, training_variables, training_target, cv=cv, scoring='recall')
    print("Average 3-Fold CV recall: {}".format(np.mean(cv_recall)))
    cv_f1 = cross_val_score(model, training_variables, training_target, cv=cv, scoring='f1')
    print("Average 3-Fold CV f1: {}".format(np.mean(cv_f1)))
    cv_precision = cross_val_score(model, training_variables, training_target, cv=cv, scoring='precision')
    print("Average 3-Fold CV precision: {}".format(np.mean(cv_precision)))


evaluation_model(pipeline_logreg(X_train, y_train, X_test, y_test), X_train, y_train)
evaluation_model(pipeline_knn(X_train, y_train, X_test, y_test), X_train, y_train)

"""
loan = loan[["NewCreditCustomer", "VerificationType", "Age", "Gender", "Country", "Amount", "Interest", "LoanDuration",
             "MonthlyPayment", "UseOfLoan", "Education", "MaritalStatus", "NrOfDependants", "EmploymentStatus",
             "EmploymentDurationCurrentEmployer", "WorkExperience", "OccupationArea", "HomeOwnershipType",
             "IncomeTotal", "LiabilitiesTotal", "DebtToIncome", "ExpectedLoss", "DefaultDate",
             "InterestAndPenaltyBalance", "AmountOfPreviousLoansBeforeLoan"]]
loan = loan.drop(["WorkExperience", "NrOfDependants", "InterestAndPenaltyBalance", "ExpectedLoss"], axis=1)
loan.info()
# Creation de la variable Default qui prend True si DefaultDate est NA False si non
# S'il y a défaut, la date du défaut est DefaultDate sinon aucune valeur n'est renseignée NA pour DefaultDate
loan["Default"] = loan["DefaultDate"].isnull()
loan["Default"] = loan["Default"].astype('str')
# Ici on remplace True par 0 pour les emprunteurs n'ayant pas fait défaut et False par 1 sinon
loan["Default"] = loan["Default"].replace("True", 0)
loan["Default"] = loan["Default"].replace("False", 1)
loan.loc[loan["Age"] < 18, "Age"] = loan['Age'].quantile(0.25)
loan['Age'] = np.where(loan['Age'] < 18, loan['Age'].quantile(0.25), loan['Age'])
loan[["AmountOfPreviousLoansBeforeLoan", "DebtToIncome"]] = loan[["AmountOfPreviousLoansBeforeLoan",
                                                                  "DebtToIncome"]].fillna(0)
cols1 = ["UseOfLoan", "MaritalStatus", "EmploymentStatus", "OccupationArea", "HomeOwnershipType"]
loan[cols1] = loan[cols1].replace({-1: 1})
loan = loan.loc[(loan["OccupationArea"] > 0) & (loan['MaritalStatus'] > 0) & (loan['EmploymentStatus'] > 0)]
loan = loan.drop(["MonthlyPayment", "DefaultDate"], axis=1)

loan['Default'].value_counts()
# Voir les valeurs manquantes
loan.isnull().sum().sort_values(ascending=False)
# Remplacer les valeurs manquantes des variables catégorielles par le mode
cols = ["VerificationType", "EmploymentDurationCurrentEmployer", "HomeOwnershipType",
        "OccupationArea", "EmploymentStatus", "MaritalStatus", "Education", "Gender"]
loan[cols] = loan[cols].fillna(loan.mode().iloc[0])

# variable AmountOfPreviousLoansBeforeLoan: Attention, pour cette variable, si des données sont manquantes cela veut
# simplement dire que l'emprunteur n'a pas de prêts antérieurs
# Donc si des données sont manquantes on les affecte la valeur 0 simplement à défaut de se reférer à DebtToIncome.
# loan.loc[loan["AmountOfPreviousLoansBeforeLoan"].isnull(),"AmountOfPreviousLoansBeforeLoan"] = 0
# loan[["AmountOfPreviousLoansBeforeLoan", "DebtToIncome"]] = loan[["AmountOfPreviousLoansBeforeLoan",
"DebtToIncome"]].fillna(0)
cols1 = ["UseOfLoan", "MaritalStatus", "EmploymentStatus", "OccupationArea", "HomeOwnershipType"]
loan[cols1] = loan[cols1].replace({-1: 1})
loan = loan.loc[(loan["OccupationArea"] > 0) & (loan['MaritalStatus'] > 0) & (loan['EmploymentStatus'] > 0)]
# Conversion des variables catégorielles en type object
categorielle = ["NewCreditCustomer", "VerificationType", "Gender", "Education",
                "EmploymentDurationCurrentEmployer", "Country", "MaritalStatus", "EmploymentStatus",
                "OccupationArea", "HomeOwnershipType", "Default", "UseOfLoan"]
for colonne in categorielle:
    loan[colonne] = loan[colonne].astype('category')
"""