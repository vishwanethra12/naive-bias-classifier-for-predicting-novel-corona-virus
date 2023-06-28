from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class Predicting_Novel_Coronavirus(models.Model):

    Fever= models.CharField(max_length=300)
    Tiredness= models.CharField(max_length=300)
    Dry_Cough= models.CharField(max_length=300)
    Difficulty_in_Breathing= models.CharField(max_length=300)
    Sore_Throat= models.CharField(max_length=300)
    None_Sympton= models.CharField(max_length=300)
    Pains= models.CharField(max_length=300)
    Nasal_Congestion= models.CharField(max_length=300)
    Runny_Nose= models.CharField(max_length=300)
    Diarrhea= models.CharField(max_length=300)
    None_Experiencing= models.CharField(max_length=300)
    Age_0To9= models.CharField(max_length=300)
    Age_10To19= models.CharField(max_length=300)
    Age_20To24= models.CharField(max_length=300)
    Age_25To59= models.CharField(max_length=300)
    Age_60Above= models.CharField(max_length=300)
    Gender_Female= models.CharField(max_length=300)
    Gender_Male= models.CharField(max_length=300)
    Gender_Transgender= models.CharField(max_length=300)
    Severity_Mild= models.CharField(max_length=300)
    Severity_Moderate= models.CharField(max_length=300)
    Severity_None= models.CharField(max_length=300)
    Contact_Dont_Know= models.CharField(max_length=300)
    Contact_No= models.CharField(max_length=300)
    Contact_Yes= models.CharField(max_length=300)
    Naive_Byes= models.CharField(max_length=300)
    SVM= models.CharField(max_length=300)
    Logistic_Regression= models.CharField(max_length=300)
    RandomForestClassifier= models.CharField(max_length=300)
    Decision_Tree_Classifier= models.CharField(max_length=300)
    KNeighborsClassifier= models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


