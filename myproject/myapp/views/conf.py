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