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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
# Importer le dataset


def importation(database):
    return pd.read_csv(database, low_memory=False)


loan = importation('C:/Users/DELL/OneDrive/Bureau/Pycharm_project/LoanData.csv')


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

loan[["AmountOfPreviousLoansBeforeLoan", "DebtToIncome"]] = loan[["AmountOfPreviousLoansBeforeLoan",
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
# Définir les variables explicatives
predictors = loan.drop('Default', axis=1)
print(predictors.shape)
predictors.info()
# Définir la variable cible
target = loan['Default']
# Diviser les variables qualitatives et quatitatives
numeric = predictors.select_dtypes(include=np.number).columns.tolist()[:-1]
categories = predictors.select_dtypes('category').columns.tolist()
print(categories)
# L'encodage des variables qualitatives et standardisation des variables quantitatives
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')
# Creation du pipeline de l'encodage et du modèle statistique
preprocessor = make_column_transformer((encoder, categories), (StandardScaler(), numeric))

# Creation des données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2)
print(X_train.shape)
print(y_train.shape)


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
         ('reducer', PCA()),
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

    # Create the GridSearchCV object: gm_cv
    gm_cv = GridSearchCV(pipeline1, parameters, cv=2)
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
          ('reducer', PCA(14)),
          ('knn', KNeighborsClassifier())]
pipe3 = Pipeline(steps3)
param_dict1 = {'knn__n_neighbors': [1, 3, 5, 7, 9, 11]}


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

    # Create the GridSearchCV object: gm_cv
    gm_cv1 = GridSearchCV(pipeline2, parameters, cv=2)
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
              ('reducer', PCA(n_components=14)),
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
              ('reducer', PCA(n_components=14)),
              ('knn', KNeighborsClassifier(n_neighbors=11))]
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

    # Compute 3-fold cross-validation scores: cv_scores
    cv_accuracy = cross_val_score(model, training_variables, training_target, cv=3, scoring='accuracy')
    print("Average 3-Fold CV accuracy: {}".format(np.mean(cv_accuracy)))
    cv_recall = cross_val_score(model, training_variables, training_target, cv=3, scoring='recall')
    print("Average 3-Fold CV recall: {}".format(np.mean(cv_recall)))
    cv_f1 = cross_val_score(model, training_variables, training_target, cv=3, scoring='f1')
    print("Average 3-Fold CV f1: {}".format(np.mean(cv_f1)))
    cv_precision = cross_val_score(model, training_variables, training_target, cv=3, scoring='precision')
    print("Average 3-Fold CV precision: {}".format(np.mean(cv_precision)))


evaluation_model(pipeline_logreg(X_train, y_train, X_test, y_test), X_train, y_train)
evaluation_model(pipeline_knn(X_train, y_train, X_test, y_test), X_train, y_train)


'''Lasso
# Penalisation: Lasso:

# Instance précise de lasso
lasso = Lasso(alpha=0.0005)
steps1 = [('one_hot', preprocessor),
          ('model', lasso)]
pipe1 = Pipeline(steps1)
# Fitting Lasso
pipe1.fit(X_train, y_train)
# Afficher coefficients
lasso_coef = pipe1['model'].coef_
print(lasso_coef)'''
