"""
Project : PD Prediction
Authors : Marwa EL GHEMMAZ, Yahia KASMI, Ismael Djoulde DIALO, Ilyass ESSBAI

"""

# Importation des Packages
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Importation des données
loan_data_frame = pd.read_csv("LoanData1.csv")

# Extraction des variables qu'on juge pertinentes pour notre étude
loan_data_frame = loan_data_frame[
    [
        "NewCreditCustomer",
        "Age",
        "Gender",
        "Country",
        "Amount",
        "Interest",
        "LoanDuration",
        "UseOfLoan",
        "Education",
        "MaritalStatus",
        "EmploymentStatus",
        "EmploymentDurationCurrentEmployer",
        "WorkExperience",
        "OccupationArea",
        "HomeOwnershipType",
        "IncomeTotal",
        "DefaultDate",
        "AmountOfPreviousLoansBeforeLoan",
    ]
]

# Création de notre variable cible "Default" depuis la variable "DefaultDate"
# Elle prend True si la valeur de DefaultDate est manquente et False sinon
# Creation of the Default variable (our target variable) that takes True
# as a value if DefaultDate is NA and False otherwise
loan_data_frame["Default"] = loan_data_frame["DefaultDate"].isnull()
loan_data_frame["Default"] = loan_data_frame["Default"].astype("str")

# Nous remplaçons True par 0 (le client est sain) et False par 1
# We then replace True by 0 meaning that the client have not defaulte and False by 1
loan_data_frame["Default"] = loan_data_frame["Default"].replace("True", 0)
loan_data_frame["Default"] = loan_data_frame["Default"].replace("False", 1)

#
# Here we drop the DefaultDate variable that is no longer needed :)
loan_data_frame = loan_data_frame.drop("DefaultDate", axis=1)
loan_data_frame["Default"].value_counts()

# Interest and DebtToIncome variables are expressed in the pourcentage format
# so we divide them by 100
loan_data_frame["Interest"] = loan_data_frame["Interest"] / 100

"""
DataVisualisation
"""


class DataVisualisation:
    def __init__(self, data):
        self.data = data

    def description(self):
        return self.data.describe()

    def missing_data(self):
        return self.data.isnull().sum()

    def boxplot(self, column):
        plt.boxplot(column, vert=True)
        plt.show()

    def correlation(self, data):
        corr = data.corr()
        ax = sns.heatmap(
            corr,
            vmin=-1,
            vmax=1,
            center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True,
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        plt.show()

    def histogram(self, column):
        sns.countplot(x=column, data=self.data, palette="hls")
        plt.xlabel(column.name)
        plt.ylabel("Fréquence")
        plt.show()

    def histogram_crosstab(self, column):
        table = pd.crosstab(column, loan_data_frame.Default)
        table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
        plt.ylabel("Défaut")
        plt.show()


"""
Preprocessing
"""
numerical_variables = loan_data_frame[
    [
        "Age",
        "Amount",
        "Interest",
        "LoanDuration",
        "IncomeTotal",
        "AmountOfPreviousLoansBeforeLoan",
    ]
]

# We begin by the description of our quantitative variables
numerical_dataviz = DataVisualisation(numerical_variables)
numerical_dataviz.description()

# We then detect the missing values
loan_dataviz = DataVisualisation(loan_data_frame)
loan_dataviz.missing_data()
# We delete the variable "WorkExperience" since the majority of its values are missing
loan_data_frame = loan_data_frame.drop("WorkExperience", axis=1)
# Replacing the missing values of our categorical variables with their respective modes
categorical_columns = [
    "EmploymentDurationCurrentEmployer",
    "HomeOwnershipType",
    "OccupationArea",
    "EmploymentStatus",
    "MaritalStatus",
    "Education",
    "Gender",
]
loan_data_frame[categorical_columns] = loan_data_frame[categorical_columns].fillna(
    loan_data_frame.mode().iloc[0]
)

# For the variable "AmountOfPreviousLoansBeforeLoan", if values are missing,
# it means that the client does not have prior loans
# So we replace the missing values with 0
loan_data_frame["AmountOfPreviousLoansBeforeLoan"] = loan_data_frame[
    "AmountOfPreviousLoansBeforeLoan"
].fillna(0)

# Extraction des valeurs uniques des variables categorilles pour detecter les valeurs aberrantes

for column in loan_data_frame:
    print(column, loan_data_frame[column].unique())

# Imputation of outliers (-1) of the variables of the list "outlier_columns" by 1
# (we consider that the error is due to typos)
# Imputation des valeurs aberrantes (-1) des variables de la liste colonnes_aberrantes
# ci-dessous par 1
# En effet, nous considérons que c'est une erreur de saisie
outlier_columns = [
    "UseOfLoan",
    "MaritalStatus",
    "EmploymentStatus",
    "OccupationArea",
    "HomeOwnershipType",
]
loan_data_frame[outlier_columns] = loan_data_frame[outlier_columns].replace({-1: 1})


# Suppression des ligne ayant des la valeur 0 pour les variables ci-dessous vu qu'elle ne figure pas parmi leur modalité
loan_data_frame = loan_data_frame.loc[
    (loan_data_frame["OccupationArea"] > 0)
    & (loan_data_frame["MaritalStatus"] > 0)
    & (loan_data_frame["EmploymentStatus"] > 0)
]

# Boxplots des variables quantitatives
for column in numerical_variables:
    print("le boxplot de la variable", column)
    loan_dataviz.boxplot(numerical_variables[column])

# Detection des valeurs aberrantes de la variable Age
loan_data_frame.loc[loan_data_frame["Age"] < 18, "Age"] = loan_data_frame[
    "Age"
].quantile(0.25)

# Détection des veleurs aberrantes au niveau de la variable
# "IncomeTotal" pour les clients qui ont un emploi
loan_data_frame.loc[
    (loan_data_frame["IncomeTotal"] <= 100) & (loan_data_frame["EmploymentStatus"] != 1)
]["IncomeTotal"].unique()

# Nous remarquons que nous avons la valeur 0
# qui figure au niveau de la variable "IncomeTotal" pour des gens employés
# Nous remplaçons alors cette valeur par le 1er quartile de la variable
loan_data_frame.loc[
    (loan_data_frame["IncomeTotal"] == 0) & (loan_data_frame["EmploymentStatus"] != 1),
    "IncomeTotal",
] = loan_data_frame["IncomeTotal"].quantile(0.25)

# Encodage des données
# On commence par transformer les modalités des variables qualitatives
# ayant comme type float en des modalités de type entier
integer_variables = [
    "Gender",
    "Education",
    "MaritalStatus",
    "EmploymentStatus",
    "OccupationArea",
    "HomeOwnershipType",
    "Default",
    "UseOfLoan",
]
for variable in integer_variables:
    loan_data_frame[variable] = loan_data_frame[variable].astype(int)

# Conversion des variables catégorielles en type categorielle
categorical_variables = [
    "NewCreditCustomer",
    "Gender",
    "Education",
    "MaritalStatus",
    "EmploymentStatus",
    "OccupationArea",
    "HomeOwnershipType",
    "Default",
    "UseOfLoan",
    "Country",
]
for variable in categorical_variables:
    loan_data_frame[variable] = loan_data_frame[variable].astype("category")

# On utilise le One-Hot-Encoding pour encoder nos variables categorielles
loan_data_frame = pd.get_dummies(
    loan_data_frame,
    columns=[
        "EmploymentDurationCurrentEmployer",
        "Country",
        "NewCreditCustomer",
        "Gender",
        "Education",
        "MaritalStatus",
        "EmploymentStatus",
        "OccupationArea",
        "HomeOwnershipType",
        "UseOfLoan",
    ],
    drop_first=False,
)
loan_data_frame

# Nous divisons notre base de données en une base d'apprentissage et une base de test
target_variable = loan_data_frame["Default"]
explanatory_variables = loan_data_frame.drop(columns=["Default"])
explanatory_train, explanatory_test, target_train, target_test = train_test_split(
    explanatory_variables, target_variable, test_size=0.3, random_state=42
)

# Vérifions si nos données équilibrées
non_default = len(loan_data_frame[loan_data_frame["Default"] == 0])
default = len(loan_data_frame[loan_data_frame["Default"] == 1])
non_default_percentage = non_default / (non_default + default)
print(
    "Le pourcentage des clients sains est", non_default_percentage * 100, "%"
)  # 66.20%
default_percentage = default / (non_default + default)
print(
    "Le pourcentage des clients défaillants est ", default_percentage * 100, "%"
)  # 33.79%


########## Unbalanced data ##############
class Resampling:
    def __init__(self, explanatory_train, target_train):
        self.explanatory_train = explanatory_train
        self.target_train = target_train

    def Unbalanced(self):
        return self.explanatory_train, self.target_train

    def Oversampling_Undersampling(self, Over=False):
        loan_actual = pd.concat([self.explanatory_train, self.target_train], axis=1)
        default = loan_actual[loan_actual["Default"] == 1]
        non_default = loan_actual[loan_actual["Default"] == 0]

        if Over == True:
            oversampled_default = resample(
                default, replace=True, n_samples=len(non_default), random_state=42
            )
            data1 = non_default
            data2 = oversampled_default
        else:
            undersampled_non_default = resample(
                non_default, replace=True, n_samples=len(default), random_state=42
            )
            data1 = default
            data2 = undersampled_non_default

        loan_new = pd.concat([data1, data2], axis=0)
        target_train = loan_new["Default"]
        explanatory_train = loan_new.drop(columns=["Default"])
        return loan_new, explanatory_train, target_train

    def SMOTE_sampling(self):
        sm = SMOTE()
        explanatory_train_sm, target_train_sm = sm.fit_resample(
            self.explanatory_train, self.target_train
        )
        loan_new = pd.concat([explanatory_train_sm, target_train_sm], axis=1)
        return loan_new, explanatory_train_sm, target_train_sm


resampled_data = Resampling(explanatory_train, target_train)
oversampled_data = resampled_data.Oversampling_Undersampling(Over=True)
undersampled_data = resampled_data.Oversampling_Undersampling(Over=False)
SMOTE_data = resampled_data.SMOTE_sampling()  # Memory problem :((( !!!


oversampled_dataviz = DataVisualisation(oversampled_data[2])
plt.title("Default histogram after oversampling")
oversampled_dataviz.histogram(oversampled_data[2])

"""
Modélisation
"""
undersampled_explanatory, undersampled_target = (
    undersampled_data[1],
    undersampled_data[2],
)

oversampled_explanatory, oversampled_target = (
    oversampled_data[1],
    oversampled_data[2],
)

decision_tree_classifier = DecisionTreeClassifier(random_state=8)
random_forest_classifier = RandomForestClassifier(random_state=8)

decision_tree_classifier.fit(oversampled_explanatory, oversampled_target)
random_forest_classifier.fit(oversampled_explanatory, oversampled_target)

###Procédure d'évaluation###
def evaluation(model):
    target_prediction = model.predict(explanatory_test)
    print("la matrice de confusion de Random forest : \n" ,  confusion_matrix(target_test, target_prediction))
    print("Le rapport de classification : \n", classification_report(target_test, target_prediction))
    print("Le roc_auc_score : \n", roc_auc_score(target_test, target_prediction))
    print("Le f1_score :\n", f1_score(target_test, target_prediction))
    print("La courbe de roc : \n")
    specificity, sensitivity,_ = roc_curve(target_test, target_prediction)
    plt.plot(specificity , sensitivity)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

evaluation(decision_tree_classifier)
evaluation(random_forest_classifier)

