from django.shortcuts import render
import csv
import pandas as pd
import numpy as np
from io import StringIO
from geopy.geocoders import Nominatim
from django.contrib import messages
from .models import *
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from django.http import HttpResponse
from .Utils import *
from django.urls import reverse

from sklearn.metrics import precision_recall_fscore_support

# Model Train Function:
global_list = []


def manual_result(request):
    return render(request, "manualresult.html")

def Home(request):
    return render(request, "index.html")


def about_us(request):
    return render(request, "about.html")

def ourteam(request):
    return render(request, 'ourteam.html')


def login_page(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        if not User.objects.filter(username=username).exists():
            messages.info(request, "Username doesnt exist.")
            return redirect("/login/")
        user = authenticate(username=username, password=password)
        if user is None:
            messages.info(request, "Invalid credentials.")
            return redirect("/login/")
        else:
            messages.info(request, "Successful.")
            user_id = request.user.id
            print(f"User ID: {user_id}")
            login(request, user)
            return redirect("/Userlog/")

    return render(request, "login.html")


def registration(request):
    if request.method == "POST":
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = User.objects.filter(username=username, password=password)
        if user.exists():
            messages.info(request, "Same username cant exist.")
            return redirect("/registration/")
        else:
            user = User.objects.create(
                first_name=first_name, last_name=last_name, username=username
            )
        user.set_password(password)
        user.save()
        messages.info(request, "Account Created Successfully")
        return redirect("/registration/")
    return render(request, "registration.html")


def manual_dataentry(request):
    if request.method == "POST":
        ccnum = request.POST.get("ccnum")
        merchant_address = request.POST.get("mer_add")
        user_address = request.POST.get("user_add")
        amount = request.POST.get("amount")
        amount = int(amount)
        calculating_distance(user_address, merchant_address)

        try:
            distance = Train_data.objects.latest("id")
            user=User.objects.latest("id")
            userId = user.id
            print("userId", userId)
        except Train_data.DoesNotExist:
            return HttpResponse("Train_data matching query does not exist.")
        
        print(distance.Lat_Distance, distance.Long_Distance)
        # Create a CSV-formatted string using the csv module
        csv_data = StringIO()
        csv_writer = csv.writer(csv_data)
        csv_writer.writerow(
            [amount, userId, distance.Lat_Distance, distance.Long_Distance]
        )

        column_names = [
            "amt",
            "user_id",
            "latitudinal_distance",
            "longitudinal_distance",
        ]
        # Get the CSV-formatted string
        csv_str = csv_data.getvalue()
        df = pd.read_csv(StringIO(csv_str), header=None, names=column_names)
        storedmodel=StoredModel_scratch.objects.latest("id")
        value =testing_manual(storedmodel, df )
        result="Not Fraud"
        if value==1:
            result="Fraud"
        
        messages.info(request, f'Result: {result}')
        return redirect(reverse('simulationResult'))

    return render(request, "manual.html")


def User_log(request):
    print(StoredModel.objects.all())
    return render(request, "Userlogged.html")


def DataTrain(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if uploaded_file:
            obj = File.objects.create(file=uploaded_file)
    try:
        df = pd.read_csv(obj.file)
    except Exception as e:
        error_message = f"Error reading the file: {str(e)}"
        return render(request, "DataTrain.html", {"error_message": error_message})
    return render(request, "DataTrain.html", {"dataframe": df})


def Trainmodel(request):
    if File.objects.exists():
        random_forest_classifer()
    else:
        return HttpResponse("error you have not imported any file to train")
    train_check = False
    if StoredModel.objects.exists():
        train_check = True
    while train_check:
        return render(request, "TrainedModel.html", {"train_check": train_check})

    return render(request, "TrainedModel.html", {"train_check": train_check})


def Testdata(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("testfile")
        if uploaded_file:
            obj = TestFile.objects.create(file=uploaded_file)
    try:
        df = pd.read_csv(obj.file)
        stored_model = StoredModel_scratch.objects.latest("id")
        testing_classifier(stored_model)
        if(testing_classifier):
            message="testing done"
            return render(request, "testdata.html",{'message':message})
    except Exception as e:
        error_message = f"Error reading the file: {str(e)}"
        return render(request, "testdata.html", {"error_message": error_message})
    return render(request, "testdata.html", {"dataframe": df})


def single_user(request):
    error_message = None  # Initialize error_message
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if uploaded_file:
            obj = SingleTrainFile.objects.create(file=uploaded_file)
            try:
                df = pd.read_csv(obj.file)
                print(df.head(5))
                return render(request, "singleusertrain.html", {"dataframe": df})
            except Exception as e:
                error_message = f"Error reading the file: {str(e)}"
                return render(
                    request, "singleusertrain.html", {"error_message": error_message}
                )

    return render(request, "singleusertrain.html")


def analytic(request):
    if StoredModel_scratch.objects.exists():
        # analysis ko thau ma eror 404 page ko html banaune
        analysis = analysisreport.objects.latest("id")
        f1_class_0 = analysis.f1_class_0
        precision_class_0 = analysis.precision_class_0
        recall_class_0 = analysis.recall_class_0
        precision_class_1 = analysis.precision_class_1
        recall_class_1 = analysis.recall_class_1
        f1_class_1 = analysis.f1_class_1
        print(f1_class_0)
        saved_conf_matrix = ConfusionMatrix.objects.last()
        print (saved_conf_matrix)

        context = {
            'saved_conf_matrix': saved_conf_matrix,
            'f1_class_0': f1_class_0,
            'precision_class_0': precision_class_0,
            'recall_class_0': recall_class_0,
            'precision_class_1': precision_class_1,
            'recall_class_1': recall_class_1,
            'f1_class_1': f1_class_1,
        }

        return render(request, "analysis.html", context)


    else:
        return HttpResponse("error you have not imported any file to train")
    
def simultaionResult(request):
    return render(request, "simulationresult.html")


def logout(request):
    clear_data = StoredModel.objects.all().delete()
    clear_analysis = analysisreport.objects.all().delete()
    clear_file = File.objects.all().delete()
    clear_testfile = TestFile.objects.all().delete()
    clear_singlefile = SingleTrainFile.objects.all().delete()
    clear_scratchmodel = StoredModel_scratch.objects.all().delete()
    print(
        clear_data,
        clear_analysis,
        clear_file,
        clear_testfile,
        clear_singlefile,
        clear_scratchmodel,
    )
    return redirect("home")

