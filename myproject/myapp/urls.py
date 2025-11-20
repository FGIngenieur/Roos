from django.urls import path
#from . import views
from . import views
from functools import partial

urlpatterns = [
    path('', views.navigation.home, name='home'),
    path('login/', views.navigation.login_view, name='login'),
    path('logout/', views.navigation.logout_view, name='logout'),

    # ✅ nouvelles pages liées à tes images
    path('r/', views.navigation.page1, name='page1'),
    path('o/', views.navigation.page2, name='page2'),
    path('RoosAI/main/', partial(views.navigation.roos_ai, tab='main'), name='roosAIMain'),
    path('RoosAI/editing/', partial(views.navigation.roos_ai, tab='editing'), name='roosAIEditing'),
    path('RoosAI/search/', partial(views.navigation.roos_ai, tab='search'), name='roosAISearch'),
    path('RoosAI/exploreQuotes/', views.navigation.list_files, name="list_files"),
    path("RoosAI/search/", views.quotes_list, name="quotes_list"),
    path("RoosAI/search/<int:quote_id>/", views.quote_detail, name="quote_detail"),
    path('s/', views.navigation.page4, name='page4'),
    path('point/', views.navigation.page5, name='page5'),
]
