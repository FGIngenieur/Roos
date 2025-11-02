from django.urls import path
from . import views
from functools import partial

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # ✅ nouvelles pages liées à tes images
    path('r/', views.page1, name='page1'),
    path('o/', views.page2, name='page2'),
    path('RoosAI/main/', partial(views.roos_ai, tab='main'), name='roosAIMain'),
    path('RoosAI/editing/', partial(views.roos_ai, tab='editing'), name='roosAIEditing'),
    path('RoosAI/search/', partial(views.roos_ai, tab='search'), name='roosAISearch'),
    path('s/', views.page4, name='page4'),
    path('point/', views.page5, name='page5'),
]
