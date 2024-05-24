# foodapp/serializers.py
from rest_framework import serializers


class Image_Upload_Serializer(serializers.Serializer):
    image = serializers.ImageField()
