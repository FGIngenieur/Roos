from .conf import *
from .libs import *
from .supabase_views import get_supabase


def import_home(request):
    return render(request, "myapp/RoosAI/import_home.html")

def blank_page(request):
    df = pd.DataFrame()
    request.session["input_data"] = df.to_json(orient="records")
    return redirect("editor-page")

def import_from_pc(request):
    if request.method == "POST":
        file: UploadedFile = request.FILES["file"]
        name = file.name

        if name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif name.lower().endswith(".xlsx"):
            df = pd.read_excel(file)
        elif name.lower().endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            df = pd.DataFrame({"pdf_content": [text]})
        else:
            return render(request, "error.html", {"msg": "Unsupported file type"})

        request.session["input_data"] = df.to_json(orient="records")
        return redirect("editor-page")

    return redirect("import-home")


def platform_list(request):
    client = get_supabase()
    response = client.table("quotesTable").select("*").execute()
    items = response.data

    return render(request, "myapp/RoosAI/modals/platform_list_modal.html", {"items": items})


def import_from_platform(request, item_id):
    client = get_supabase()
    row = client.table("quotesTable").select("*").eq("id", item_id).single().execute().data
    file_bytes = client.storage.from_(row["bucket"]).download(row["file_path"])

    name = row["file_name"].lower()

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(file_bytes))
    elif name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        df = pd.DataFrame({"pdf_content": [text]})

    request.session["input_data"] = df.to_json(orient="records")

    # HTMX-friendly response:
    return HttpResponse(f"""
        <script>
            window.location.href = '/RoosAI/editor/';
        </script>
    """)

def editor_page(request):
    import json
    rows_json = request.session.get("input_data", "[]")
    rows = json.loads(rows_json)

    # Pagination
    page = int(request.GET.get("page", 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page

    page_rows = rows[start:end]
    has_next = end < len(rows)
    has_prev = page > 1

    ctx = {
        "rows": page_rows,
        "page": page,
        "has_next": has_next,
        "has_prev": has_prev,
    }
    return render(request, "myapp/RoosAI/editor_page.html", ctx)
