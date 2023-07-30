
#Importation des donnees brutes
donnees<-data.frame(read.csv("C:/Users/Acer/Desktop/M1 SEP/S1 M1 SEP/LoanData.csv",sep =",",  header=TRUE))

#Affichage du nombre d'observations de donnees
length(row.names(donnees))

#affichage du nombre de variables (colonnes) de donnees
length(colnames(donnees))

#Exportation des 20 premieres observations des donnees brutes sur notre repertoire au nom de Bondora_donnees_brutes
write.table(head(donnees, n=20),"Bondora_donnees_brutes.csv",sep=";", row.names = F)

View(donnees)
attach(donnees)
#VerificationType, 
#Method used for loan application data verification 
#0 Not set 1 Income unverified 2 Income unverified, cross-referenced by phone 
#3 Income verified 4 Income and expenses verified


#Nous ne retenons que des observations dont les donnees sur les revenus et les depenses ont ete authetifie�es par Bondora que , nous excluons donc toutes les donnees non verifiées et les emprunteurs ayant plus de 60 ans

donnees_clea<-donnees[donnees$VerificationType==4 & donnees$Age<=60,]
donnees_clea<-donnees_clea[,sort(names(donnees_clea))]
#l'emprunteur exprime le montant de l emprunt et accepte parmi ceux qui lui sont proposes par bondora celui qui minimise ses charges d'interets

attach(donnees_clea)
#nous retenons les categories (3,4 et 5) pour l education(2,3,4 et 5) pour employmentstatus et (1,2,3,4,5,6,7,8,9) pour homeOwnerShipType
donnees_clea<-donnees_clea[donnees_clea$Education %in% c(3,4,5) & donnees_clea$EmploymentStatus %in% c(2,3,4,5) & donnees_clea$HomeOwnershipType %in% c(1,2,3,4,5,6,7,8,9) ,]

#Nous allons nous restreindre aux variables existantes avant l'octroi definitif du pret et non celles generees pour suivre le pret jusqu'a son echeance
#Nous aurions pu ne retenir que les variables qui nous interessent au lieu d'exclure celles qui nous ne intersent pas
donnees_clean<-subset(donnees_clea, select=-c(LoanNumber, DefaultDate, VerificationType,LoanId, ListedOnUTC, PartyId, ApplicationSignedHour, 
                           ApplicationSignedWeekday, LanguageCode, DateOfBirth, Gender, 
                           Country, County, City, EmploymentPosition,
                           OccupationArea, HomeOwnershipType, IncomeFromChildSupport,Education, InterestAndPenaltyPaymentsMade, 
                           IncomeFromFamilyAllowance, IncomeFromLeavePay, IncomeFromPrincipalEmployer, 
                           IncomeFromPension, IncomeFromSocialWelfare, ContractEndDate, FirstPaymentDate, 
                           MaturityDate_Original, MaritalStatus, ActiveLateCategory,ActiveLateLastPaymentCategory, 
                           ActiveScheduleFirstPaymentReached, BiddingStartedOn, BidsApi, BidsManual, BidsPortfolioManager, 
                           CreditScoreEeMini, CreditScoreEsEquifaxRisk, CreditScoreEsMicroL, CreditScoreFiAsiakasTietoRiskGrade, 
                           CurrentDebtDaysPrimary, CurrentDebtDaysSecondary, DebtOccuredOn, DebtOccuredOnForSecondary, EAD1, EAD2,
                           EL_V0, EL_V1, FreeCash, GracePeriodEnd, GracePeriodStart, Rating_V0, Rating_V1, Rating_V2, RecoveryStage, 
                           RefinanceLiabilities, ReScheduledOn, Restructured, StageActiveSince, WorseLateCategory, 
                           IncomeOther, LastPaymentOn, LoanApplicationStartedDate, InterestAndPenaltyBalance,
                           LoanDate, MaturityDate_Last, MonthlyPayment, ModelVersion, NextPaymentDate,NextPaymentNr, NewCreditCustomer,
                           NrOfScheduledPayments, PlannedPrincipalTillDate, PlannedPrincipalPostDefault, PreviousEarlyRepaymentsBefoleLoan, PreviousRepaymentsBeforeLoan,
                           PrincipalBalance, PrincipalDebtServicingCost, PrincipalOverdueBySchedule, PrincipalPaymentsMade, PrincipalRecovery, PrincipalWriteOffs,
                           InterestAndPenaltyDebtServicingCost, InterestAndPenaltyWriteOffs, InterestRecovery,
                           LiabilitiesTotal, ExistingLiabilities, MonthlyPaymentDay, NoOfPreviousLoansBeforeLoan,
                           Rating, ReportAsOfEOD, PreviousEarlyRepaymentsCountBeforeLoan, PlannedInterestTillDate,
                           PlannedInterestPostDefault, EmploymentDurationCurrentEmployer, Status, WorkExperience
                           
                           ) )

View(donnees_clean)
archo<-for (ik in colnames(donnees_clean)) {paste(",",ik)}
# Exclusion des observations n'ayant pas d'information precise sur le nombre de personnes dependantes
condition=as.character(c(0,1,2,3,4,5,6,7,8,10))

donnees_cleane<-donnees_clean[donnees_clean$NrOfDependants %in% condition,]

donnees_cleane$NrOfDependants=as.numeric(donnees_cleane$NrOfDependants)

View(donnees_cleane)

#Exclusions des observations n'ayant pas de valeurs definies pour la  variable LosseGivenDefault
cond2<-is.na(donnees_cleane$LossGivenDefault)

donnees_cleane1<-donnees_cleane[donnees_cleane$LossGivenDefault !=cond2,]
ins
#Nous gardons finalement que les observations ayant des informations completes
donnees_cleane2=donnees_cleane1[complete.cases(donnees_cleane1),]

# Derniere conversion du jeu de donnees en dataframe avant export
donnees_cleaned<-as.data.frame(donnees_cleane2)


donnees_cleaned$Interest<-donnees_cleaned$Interest*0.01


donnees_cleaned$DebtToIncome<-donnees_cleaned$DebtToIncome*0.01
#Nombres d'observations
length(row.names(donnees_cleaned))

#Nombre de variables
length(donnees_cleaned)

#Sauvegarde du jeu de donnees nettoye pour notre etude sur notre machine
write.table(donnees_cleaned,"Bondora_Cleaned_data.csv",sep=";", row.names = F)

#Visualisation des 25 premieres observations
View(head(donnees_cleaned, n=25))

## IV-	Analyse exploratoire des donnees retenues pour le projet
detach(donnees) #les dataframes ont les memes variiables, nous devons donc les detachees pour ne pas que R les confondes
detach(donnees_clea) #les dataframes ont les memes variiables, nous devons donc les detachees pour ne pas que R les confondes
attach(donnees_cleaned)# R n'aura acces direct qu'aux variables de ce dataframe

#Statistiques descriptives
install.packages("Hmisc")
library(Hmisc)
description_complete<-describe(donnees_cleaned)
description_complete_list<-as.list(description_complete)
description_complete_list[1]


install.packages("vtable")#Pour pour avoir exporter les sorties des stats descriptives au format CSV
library(vtable)
#fichier contenant quelques statistiques
sumtable(donnees_cleaned, out='csv', file="Statistiques_descriptives.csv")

library(ggplot2) # Pour la creation de graphique
library(tidyverse) 

#hISTOGRAM de la distribution Amount

Visualisation_de_Amount <- donnees_cleaned %>%
  
  ggplot( aes(x=Amount)) +
  geom_histogram( fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  stat_function(fun = dnorm,
                args = list(mean = mean(Amount),
                            sd = sd(Amount)),
                col = "#1b98e0",
                size = 5)+
  ggtitle("Visualisation de Amount") +
  
  theme(plot.title = element_text(size=15, hjust = 0.5)
  )

X11() #Visualisation du garphique dans une autre fenetre
Visualisation_de_Amount

#Diagramme en barre de la distribution des emprunts selon leur motif

Variable_UseOFloan<-as.factor(UseOfLoan)
Visualisation_de_UseOfLoan <- donnees_cleaned %>%
  
  ggplot( aes(x=UseOfLoan)) +
  geom_bar( fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  ggtitle("Visualisation de UseOfLoan") +

  theme(plot.title = element_text(size=15, hjust = 0.5)
  )

X11() #Visualisation du garphique dans une autre fenetre
Visualisation_de_UseOfLoan

#Diagramme en barre de la distribution de l'age des emprunteurs
Variable_Age<-as.factor(Age)
pAge <- donnees_cleaned %>%
  
  ggplot( aes(x=Variable_Age)) +
  geom_bar( fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  ggtitle("Visualisation de l'Age") +
  
  theme(
    plot.title = element_text(size=15, hjust = 0.5)
  )

X11() #Visualisation du garphique dans une autre fenetre
pAge


#Representation graphique avec une boite a moustaches 'diagramme de Tukey' de  la distribution de Amount en fonction de UseOfLoan
Variable_UseOfLoan<-as.factor(UseOfLoan)
UseOfLoan_Amount<-ggplot(donnees_cleaned, aes(x=Variable_UseOfLoan, y=Amount)) + 
  stat_boxplot(geom = "errorbar", width = 0.15, color = 1) +
                     geom_boxplot(fill='#e9ecef')

medianAmoun4<-donnees_cleaned[UseOfLoan==4,]
median(medianAmoun4$Amount)
X11()
UseOfLoan_Amount

#Representation graphique avec des boites a moustaches de  la distribution de NrOfDependants vs Amount
Variable_NrOfDependants<-as.factor(NrOfDependants)
NrOfDependants_Amount<-ggplot(donnees_cleaned, aes(x=Variable_NrOfDependants, y=Amount)) +
  stat_boxplot(geom = "errorbar", width = 0.15, color = 1)+  geom_boxplot(fill='#e9ecef')
X11()
NrOfDependants_Amount

medianAmoun5<-donnees_cleaned[NrOfDependants==1,]
describe(medianAmoun5$Amount)
#

#Representation graphique de  la distribution de EmploymentStatus vs Amount
Variable_EmploymentStatus<-as.factor(EmploymentStatus)
EmploymentStatus_Amount<-ggplot(donnees_cleaned, aes(x=Variable_EmploymentStatus, y=Amount)) +stat_boxplot(geom = "errorbar", width = 0.15, color = 1)  +geom_boxplot(fill='#e9ecef')
X11()
EmploymentStatus_Amount
# Distribution de Amount vs AppliedAmount
color_EmploymentStatus<-as.factor(EmploymentStatus)
AppliedAmount_Amount<-ggplot(donnees_cleaned, aes(x=AppliedAmount, y=Amount ))+
  geom_point(aes(color =color_EmploymentStatus , size = ProbabilityOfDefault), alpha = 0.5)

table(EmploymentStatus) # Repartition des emprunteurs selon le statut de leur emploi i.e modalité d'EmploymentStatus
X11()
AppliedAmount_Amount
#test de Student de AppliedAmount versus Amount
t.test(Amount, AppliedAmount)
# distribution de Amount vs ExpectedReturn
color_LoanDuration<-as.factor(LoanDuration)
ExpectedReturn_Amount<-ggplot(donnees_cleaned, aes(x=ExpectedReturn, y=Amount ))+
  geom_point(aes(color =color_LoanDuration , size = ProbabilityOfDefault), alpha = 0.5)

X11()
ExpectedReturn_Amount



# Exploration de la nature de la coorelation dans le jeu de donnees
install.packages("PerformanceAnalytics")
library("PerformanceAnalytics")

X11()
chart.Correlation(donnees_cleaned, pch=1, method = "pearson")

#V-	Modelisation de la variable Amount
#creation des echatillons donnees_train donnees_test

taille <- round(nrow(donnees_cleaned) * 0.70)
donnees_train<-donnees_cleaned[1:taille, ]
write.table(donnees_train,"donnees_train.csv",sep=";", row.names = F)

donnees_test <-donnees_cleaned[(taille + 1):nrow(donnees_cleaned), ]
write.table(donnees_test,"donnees_test.csv",sep=";", row.names = F)

# Exploration de la variable a explique sur donnees_train
# Modelisation de Amount
detach(donnees_cleaned)
attach(donnees_train)
detach(donnees_train)
detach(donnees_cleaned)
data<-donnees_cleaned
data$EmploymentStatus<-as.factor(data$EmploymentStatus)
data$UseOfLoan<-as.factor(data$UseOfLoan)

attach(data)
str(donnees_cleaned)
modele<-lm(Amount ~., donnees_train)
library("glmulti")
library("boot")

library(glmulti)
Modele.genetique<-glmulti(Amount ~.,data=data, level = 1, method = "g", 
                          fitfunction = lm, crit = 'aic', plotty = F )
Modele.genetiqueAIC<-Modele.genetique
Modele.genetiqueAIC.best.model <- summary(Modele.genetiqueAIC)$bestmodel
Modele.genetiqueAIC.best.model
summary(Modele.genetiqueAIC.best.model)
Modele.genetiqueAIC.best.model$coefficients[1:15]
Modele.genetiqueAIC.best.model$finalModel


plot(modele)

# Modele2 
modele2 <- lm(Amount ~ Age+ AmountOfPreviousLoansBeforeLoan+ 
                     AppliedAmount+ DebtToIncome+ProbabilityOfDefault+ UseOfLoan , donnees_train)

summary(modele2)

#
modele_final <- lm(Amount ~AppliedAmount+ DebtToIncome+ProbabilityOfDefault+ UseOfLoan , donnees_train)


summary(modele_final)

modele_final$coefficients[5]

#plot(modele_final)
###
install.packages("car")

library(car) 
#Calcul de la statistique de durbinWatsonTest
durbinWatsonTest(modele_final)

# Verification de l'existence ou non de la multicolinearite des var dans la regression
vif(modele_final)
# Prediction
prediction<-predict(modele_final, newdata =donnees_test)
prediction
summary(prediction)


X11()
plot(donnees_test$Amount, prediction, ylab="Valeurs predites de Amount", xlab="Valeurs de Amount dans donnees_test", col="red")

prediction_IC<-predict(modele_final, newdata =donnees_test,interval="confidence" )
prediction_IC
write.table(as.data.frame(prediction_IC),"prediction_IC.csv",sep=";", row.names = F)
summary(prediction_IC)
#CONCLUSION
vcc<-as.data.frame(prediction_IC)
cbind.data.frame(vcc, donnees_test$Amount)
Emprunts_<donnees_test$Amount
head(cbind.data.frame(vcc, donnees_test$Amount))
attach(vcc)
sum(vcc$upr)
sum(vcc$lwr)

1.5*sum(vcc$upr)
1.5*sum(vcc$lwr)
