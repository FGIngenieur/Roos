from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # ✅ nouvelles pages liées à tes images
    path('r/', views.page1, name='page1'),
    path('o/', views.page2, name='page2'),
    path('o2/', views.page3, name='page3'),
    path('s/', views.page4, name='page4'),
    path('point/', views.page5, name='page5'),
]
