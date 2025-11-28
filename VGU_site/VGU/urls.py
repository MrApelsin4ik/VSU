# urls.py
from django.urls import path, include
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from main_app.views import *

urlpatterns = [
    path("", main_page, name="main_page"),
    path("search/", search, name="search"),
    path("news/<int:pk>/", news_detail, name="news_detail"),
    path("announcement/<int:pk>/", announcement_detail, name="announcement_detail"),




    path("admin/news/create/", admin_news_create, name="admin_news_create"),
    path("admin/login/", admin_login, name="admin_login"),
    path("admin/main/", admin_main, name="admin_main"),
    path("admin/news/", admin_news_list, name="admin_news_list"),
    path("admin/announcement/", admin_announcement_list, name="admin_announcement_list"),
    path("admin/courses/", admin_course_list, name="admin_course_list"),
    path("admin/sections/", admin_section_list, name="admin_section_list"),

    path("admin/", admin.site.urls),


    path('ckeditor/', include('ckeditor_uploader.urls')),




    path('api/ai/missing/', MissingAIListView.as_view(), name='ai-missing'),
    path('api/ai/update/', UpdateAIView.as_view(), name='ai-update'),
    path('api/ai/full/<str:model>/<int:pk>/', FullObjectView.as_view(), name='ai-full'),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)