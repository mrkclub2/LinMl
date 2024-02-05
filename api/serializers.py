from rest_framework.serializers import ModelSerializer,Serializer


class LoginSerializer(ModelSerializer):
    class Meta:
        model = ''
        fields = ''

    def validate(self, attrs):
        pass
