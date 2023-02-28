from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('cal', views.cal),
    path('chart', views.getChart)
]

