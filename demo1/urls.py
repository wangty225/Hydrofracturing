from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('getPage2', views.getPage2),
    path('2', views.index2),

]

