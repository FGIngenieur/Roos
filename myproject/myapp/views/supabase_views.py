from .conf import *
from .libs import *

@login_required(login_url='/login/')
def serve_supabase_file(request, filename):
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    res = client.storage.from_(settings.SUPABASE_STORAGE_BUCKET).download(filename)

    if not res or not res.get("data"):
        return HttpResponseForbidden("File not found or unauthorized")

    return FileResponse(BytesIO(res["data"]), as_attachment=False, filename=filename)

def get_supabase():
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
