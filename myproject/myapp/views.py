from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

# 🏠 Page d'accueil (accessible uniquement si connecté)
@login_required(login_url='/login/')
def home(request):
    return render(request, 'myapp/home.html')

# 🔐 Page de connexion
def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # redirige vers la page d'accueil
        else:
            # Mauvais identifiants → on réaffiche le formulaire avec un message d'erreur
            return render(request, 'myapp/login.html', {'error': 'Identifiants invalides'})
    
    # Requête GET → on affiche le formulaire
    return render(request, 'myapp/login.html')

def page1(request):
    return render(request, 'myapp/page1.html')

def page2(request):
    return render(request, 'myapp/page2.html')

def page3(request):
    return render(request, 'myapp/page3.html')

def page4(request):
    return render(request, 'myapp/page4.html')

def page5(request):
    return render(request, 'myapp/page5.html')

# 🚪 Page de déconnexion
def logout_view(request):
    logout(request)
    return redirect('login')  # redirige vers la page de connexion