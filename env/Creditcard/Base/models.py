from django.db import models
from django.contrib.auth.models import User
from django.db import models
from joblib import dump, load
from io import BytesIO


class analysisreport(models.Model):
    accuracy = models.FloatField(null=True, blank=True)
    precision_class_0 = models.FloatField(null=True, blank=True)
    recall_class_0 = models.FloatField(null=True, blank=True)
    f1_class_0 = models.FloatField(null=True, blank=True)
    precision_class_1 = models.FloatField(null=True, blank=True)
    recall_class_1 = models.FloatField(null=True, blank=True)
    f1_class_1 = models.FloatField(null=True, blank=True)

class evaluationMetrics(models.Model):
    accuracy = models.FloatField(null=True, blank=True)
    precision_class_0 = models.FloatField(null=True, blank=True)
    recall_class_0 = models.FloatField(null=True, blank=True)
    f1_class_0 = models.FloatField(null=True, blank=True)
    precision_class_1 = models.FloatField(null=True, blank=True)
    recall_class_1 = models.FloatField(null=True, blank=True)
    f1_class_1 = models.FloatField(null=True, blank=True)


class StoredModel(models.Model):
    serialized_model = models.BinaryField()

    @property
    def model(self):
        return load(BytesIO(self.serialized_model))

    @model.setter
    def model(self, value):
        bio = BytesIO()
        dump(value, bio)
        self.serialized_model = bio.getvalue()


class StoredModel_scratch(models.Model):
    serialized_model = models.BinaryField()

    @property
    def model(self):
        return load(BytesIO(self.serialized_model))

    @model.setter
    def model(self, value):
        bio = BytesIO()
        dump(value, bio)
        self.serialized_model = bio.getvalue()


# Create your models here.
class Account(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)


class Train_data(models.Model):
    Lat_Distance = models.FloatField(max_length=10)
    Long_Distance = models.FloatField(max_length=10)


class File(models.Model):
    file = models.FileField(upload_to="files")


class TestFile(models.Model):
    file = models.FileField(upload_to="testfiles")


class SingleTrainFile(models.Model):
    file = models.FileField(upload_to="Singlefiles")

class ConfusionMatrix(models.Model):
    true_positive = models.IntegerField()
    true_negative = models.IntegerField()
    false_positive = models.IntegerField()
    false_negative = models.IntegerField()
