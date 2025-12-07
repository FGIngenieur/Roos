# libs.py — central place for imports

import os
import io
from io import BytesIO
import pandas as pd
import pdfplumber

from django.shortcuts import render, redirect
from django.http import (
    HttpResponse,
    JsonResponse,
    FileResponse,
    HttpResponseForbidden,
)
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

from django.conf import settings
from django.utils import timezone
from django.core.files.uploadedfile import UploadedFile

from supabase import create_client
from django.urls import reverse
