from django.urls import path
from . import views

urlpatterns = [
    path('', views.base),
    path(r'download', views.download,name='download'),
    path(r'calculate', views.calculate,name='calculate'),

]

