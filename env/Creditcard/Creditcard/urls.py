from django.contrib import admin
from django.urls import path
from Base.views import *
from django.conf import settings
from django.conf.urls.static import static
from Base.templates import *

urlpatterns = [
    path("logout/", logout, name="logout"),
    path("about/", about_us, name="about"),
    path("", Home, name="home"),
    path("login/", login_page, name="login"),
    path("manual_result/", manual_dataentry, name="manual_dataentry"),
    path("registration/", registration, name="registration"),
    path("Userlog/analytic", analytic, name="analytic"),
    path("Userlog/singleuser", single_user, name="single_user"),
    path("Userlog/", User_log, name="Userlog"),
    path("Testdata/", Testdata, name="Testdata"),
    path("Datatrain/Modeltrain/", Trainmodel, name="Datatrain"),
    path("analytic/", analytic, name="analytic"),
    path("Datatrain/", DataTrain, name="Datatrain"),
    path("Dashboard/", Dashboard, name="Dashboard"),
    path("admin/", admin.site.urls),
]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
