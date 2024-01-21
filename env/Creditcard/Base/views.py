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

from sklearn.metrics import precision_recall_fscore_support

# Model Train Function:
global_list = []


def manual_result(request):
    return render(request, "manualresult.html")

    # def calculating_distance(args1, args2):
    #     loc = Nominatim(user_agent="Geopy Library", timeout=None)
    #     getLoc1 = loc.geocode(args1)
    #     getLoc2 = loc.geocode(args2)
    #     print(getLoc1.address)
    #     print(getLoc2.address)
    #     print("Latitude 1 = ", getLoc1.latitude, "\n")
    #     print("Longitude 1 = ", getLoc1.longitude)
    #     print("Latitude 2 = ", getLoc2.latitude, "\n")
    #     print("Longitude 2 = ", getLoc2.longitude)
    #     latitudinal_distance = abs(round(getLoc1.latitude - getLoc2.latitude, 3))
    #     longitudinal_distance = abs(round(getLoc1.longitude - getLoc2.longitude, 3))
    #     distance = Train_data()
    #     distance.Lat_Distance = longitudinal_distance
    #     distance.Long_Distance = latitudinal_distance
    #     distance.save()

    if File.objects.exists():
        dataread = File.objects.latest("id")
        df = pd.read_csv(dataread.file)
        X_train = df.drop("is_fraud", axis=1)
        y_train = df["is_fraud"]
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        stored_model = StoredModel.objects.create(model=clf)
        stored_model.save()
        y_pred1 = clf.predict(X_train)
        global_list.append(y_pred1)
        return stored_model

    else:
        return HttpResponse("no file to train")


# Create your views here.
def Home(request):
    return render(request, "index.html")


def about_us(request):
    random_forest_classifer()
    storemodel = StoredModel_scratch.objects.latest("id")
    testing_classifier(storemodel)
    return render(request, "about.html")


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
        print(ccnum, merchant_address, amount, user_address)
        print(merchant_address, user_address)
        print(calculating_distance(user_address, merchant_address))
        try:
            distance = Train_data.objects.latest("id")
            userId = distance.id
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
        print(df)

        stored_model = StoredModel.objects.latest("id")
        # manual_testing(stored_model, df)

    return render(request, "manual.html")


def User_log(request):
    print(StoredModel.objects.all())
    return render(request, "Userlogged.html")


def Reports(request):
    redirect("/Reports")
    return render(request, "report.html")


def Dashboard(request):
    calculating_distance()
    return render(request, "dashboard.html")


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
    if not StoredModel_scratch.objects.exists():
        # analysis ko thau ma eror 404 page ko html banaune
        return render(request, "analysis.html", {"message": "model not trained"})

    else:
        # Assuming you have the test data available in the request or retrieve it from the database
        stored_model = StoredModel_scratch.objects.latest("id")
        testing_classifier(stored_model)

        analysis = analysisreport.objects.latest("id")
        # accuracy = analysis.accuracy
        # print(accuracy)
        f1_class_0 = analysis.f1_class_0
        precision_class_0 = analysis.precision_class_0
        recall_class_0 = analysis.recall_class_0
        precision_class_1 = analysis.precision_class_1
        recall_class_1 = analysis.recall_class_1
        f1_class_1 = analysis.f1_class_1
        return render(request, "analysis.html")


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
