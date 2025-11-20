from .conf import *

if settings.DEV_FLAG:
    user_id = int(settings.USER_ID)
    project_id = 2
    client_id = 1

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


def record_file_upload(file_name : str,
 client ,
  table = "quotesTable",
  bucket="dev-kev",
  folder="quotesImported",
  user_id = user_id,
  project_id = project_id,
  client_id = client_id):
    try:
        response = client.table(table).insert({
            "file_name": file_name,
            "file_path": f'{folder}/{file_name}',
            "bucket"  : bucket,
            "project_id" : project_id,
            "client_id" : client_id,
            "author_id" : user_id
        }).execute()
        print(response.status_code, response.data)
        # Check status
        if response.status_code != 201:
            print("Insert failed:", response.data)
            return False
        
        print("File recorded successfully:", response.data)
        return True
    except Exception as e:
        print("Error recording file:", e)
        return False


def list_files(request, table = "quotesTable", user_id = user_id):
    try:
        # Fetch all rows from uploaded_files
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        response = supabase.table(table).select("*").eq("author_id", user_id).execute()
        data = response.data  # ✅ the returned rows

        # Check if data is None or empty
        if data is None:
            return JsonResponse({"files": []})

        return JsonResponse({"files": data})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)


def quotes_list(request, tab, client, user_id = user_id):
    client_id = client.table("myapp_profile").select("client_id").eq("user_id", user_id).execute().data or []
    # Fetch all quotes
    res = client.table("quotesTable").select("*").eq("client_id", client_id).execute()
    quotes = res.data or []

    return render(request, f"myapp/RoosAI/{tab}.html", {"quotes": quotes})

def quote_detail(request, quote_id, client, user_id):
    res = client.table("quotes").select("*").eq("id", quote_id).single().execute()
    quote = res.data
    return render(request, f"myapp/RoosAI/quote_detail.html", {"quote": quote})


@login_required(login_url='/login/')
def roos_ai(request, tab = 'main'):
    if tab == 'main':
        return render(request, f'myapp/RoosAI/{tab}.html')
    elif tab == 'editing':
        text = None
        print(request.FILES)
        if request.method == 'POST' and 'importFile' in request.FILES:
            uploaded_file = request.FILES['importFile']

            if uploaded_file.size > 2 * 1024 * 1024:
                return render(request, 'myapp/RoosAI/editing.html', {'text': 'File too large!'})
            if not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx')):
                return render(request, 'myapp/RoosAI/editing.html', {'text': 'Only .csv and .xlsx files allowed!'})
            
            supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            bucket = settings.SUPABASE_STORAGE_BUCKET

            # Upload file to Supabase Storage
            #res = supabase.storage.from_(bucket).upload(uploaded_file.name, uploaded_file.read(), {"upsert": True})
            #print(f"This is status code : {res.status_code}")
            
            record_file_upload(uploaded_file.name, supabase)
                
            #else:
              #  print("Storage upload failed:", res.data)

            #if res.get("error"):
             #   return HttpResponse(f"Upload failed: {res['error']}", status=400)

            #return HttpResponse("✅ File uploaded successfully!")
        return render(request, 'myapp/RoosAI/editing.html', {
        "SUPABASE_URL": settings.SUPABASE_URL,
        "SUPABASE_KEY": settings.SUPABASE_KEY,
        "SUPABASE_STORAGE_BUCKET" : settings.SUPABASE_STORAGE_BUCKET,
    })
    elif tab == 'search':
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        quotes_list(request, tab, supabase)
        


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