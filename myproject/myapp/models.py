from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar = models.ImageField(upload_to='avatars/', default='avatars/default.png', blank=True)

    def __str__(self):
        return self.user.username

class Document(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="docs/")
