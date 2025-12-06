from django.core.files.storage import Storage
from django.core.files.base import ContentFile
from django.conf import settings
from supabase import create_client

class SupabaseStorage(Storage):
    """
    A custom Django storage backend for Supabase Storage.
    Supports both public and private buckets.
    """

    def __init__(self):
        self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        self.bucket = getattr(settings, "SUPABASE_STORAGE_BUCKET", "dev-kev")
        self.public = getattr(settings, "SUPABASE_STORAGE_PUBLIC", True)

    def _save(self, name, content):
        """Upload file to Supabase Storage"""
        data = content.read()
        # Upload file (overwrites if exists)
        self.client.storage.from_(self.bucket).upload(name, data, {"upsert": True})
        return name

    def _open(self, name, mode="rb"):
        """Download file content from Supabase"""
        res = self.client.storage.from_(self.bucket).download(name)
        if not res or not res.get("data"):
            raise FileNotFoundError(f"File '{name}' not found in Supabase bucket '{self.bucket}'")
        return ContentFile(res["data"])

    def url(self, name):
        """Return a public or signed URL depending on settings"""
        bucket = self.client.storage.from_(self.bucket)

        if self.public:
            # Public bucket: direct public URL
            return bucket.get_public_url(name)
        else:
            # Private bucket: generate a signed URL (default 1 hour)
            signed = bucket.create_signed_url(name, expires_in=3600)
            return signed.get("signedURL")

    def delete(self, name):
        """Delete a file from storage"""
        self.client.storage.from_(self.bucket).remove([name])

    def exists(self, name):
        """Check if a file exists in storage"""
        try:
            res = self.client.storage.from_(self.bucket).list()
            return any(f["name"] == name for f in res.get("data", []))
        except Exception:
            return False
