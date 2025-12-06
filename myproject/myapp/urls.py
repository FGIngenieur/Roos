from django.urls import path
from . import views
from .views import create_quotes
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
    path("RoosAI/search/", views.navigation.quotes_list, name="quotes_list"),
    path("RoosAI/search/<int:quote_id>/", views.navigation.quote_detail, name="quote_detail"),
    path("RoosAI/import_home/", create_quotes.import_home, name="import-home"),
    path("RoosAI/upload/", create_quotes.import_from_pc, name="import-from-pc"),
    path("RoosAI/platform/", create_quotes.platform_list, name="platform-list"),
    path("RoosAI/platform/import/<int:item_id>/", create_quotes.import_from_platform, name="import-from-platform"),
    path("RoosAI/blank/", create_quotes.blank_page, name="blank-page"),
    path("RoosAI/editor/", create_quotes.editor_page, name="editor-page"),
]
