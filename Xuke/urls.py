from django.urls import path
from . import views

urlpatterns = [
    path('', views.prod),
    path('prod/login/', views.prod_login),
    path('prod/home/', views.prod_home),
    path('prod/open/', views.prod_open),
    path('prod/grey/', views.prod_grey),
    path('prod/Lasso/', views.prod_Lasso),
]
