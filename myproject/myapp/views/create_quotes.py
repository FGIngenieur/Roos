from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, HttpResponseForbidden, HttpResponse
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from supabase import create_client
from django.conf import settings
from django.utils import timezone
from io import BytesIO
import os
#import backends
#from backends import (
#    quotes, quoteMapping, quotationEngine, quoteClustering, dataLoader
#)
import pandas as pd
import io
import pdfplumber
from django.core.files.uploadedfile import UploadedFile


def import_home(request):
    return render(request, "myapp/RoosAI/import_home.html")

def blank_page(request):
    df = pd.DataFrame()
    request.session["input_data"] = df.to_json(orient="records")
    return redirect("myapp/RoosAI/editor-page")

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
        return redirect("myapp/RoosAI/editor-page")

    return redirect("myapp/RoosAI/import-home")


def platform_list(request):
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    response = supabase.table("quotesTable").select("*").execute()
    items = response.data

    return render(request, "myapp/RoosAI/modals/platform_list_modal.html", {"items": items})

from django.http import HttpResponse

def import_from_platform(request, item_id):
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    row = supabase.table("quotesTable").select("*").eq("id", item_id).single().execute().data
    file_bytes = supabase.storage.from_(row["bucket"]).download(row["file_path"])

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
    return HttpResponse("""
        <script>
            window.location.href = 'RoosAI/editor/';
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
